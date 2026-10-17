import gc
import unittest
from dataclasses import replace
from unittest.mock import patch
import os
import numpy as np
import scipy.sparse as sp
from parsec_python.acceleration.backends.cupy import cupy_available,CuPyHamiltonian,require_cupy
from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter,FilterBlock
from parsec_python.acceleration.Eigensolvers.distributed_filter import block_partitions
from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState,run_subspace_filter,SubspaceSettings

class PartitionTests(unittest.TestCase):
    def test_symmetric_condition_screen_retains_svd_near_guard(self):
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import _overlap_condition
        rng=np.random.default_rng(93);q=np.linalg.qr(rng.normal(size=(35,35)))[0]
        with patch.dict(os.environ,PARSEC_CUPY_RITZ_CONDITION='symmetric',PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8'):
            for condition in (1.,1e3,1e5,1e7,1e9):
                a=(q*np.geomspace(1,condition,35))@q.T
                expected=np.linalg.cond(a)
                with patch('numpy.linalg.cond',wraps=np.linalg.cond) as svd:
                    actual=_overlap_condition(a)
                    self.assertEqual(svd.call_count,int(condition>=1e6))
                self.assertAlmostEqual(actual/expected,1.,delta=2e-10)
            with patch('numpy.linalg.cond',wraps=np.linalg.cond) as svd:
                _overlap_condition(np.diag([-1.,2.,3.]))
                svd.assert_called_once()

    def test_slab_and_tile_share_one_buffer_on_the_host(self):
        import importlib
        from types import SimpleNamespace
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        # NumPy stands in for CuPy; a host array takes its memory as ``buffer`` instead of ``memptr``.
        shim=SimpleNamespace(float64=np.float64,dtype=np.dtype,empty=np.empty,zeros=np.zeros,
                             asfortranarray=np.asfortranarray,ascontiguousarray=np.ascontiguousarray,
                             ndarray=lambda shape,dtype,memptr,order:np.ndarray(shape,dtype=dtype,buffer=memptr,order=order))
        rng=np.random.default_rng(67);operator=rng.normal(size=(61,61));operator+=operator.T
        host=rng.normal(size=(61,7));c=rng.normal(size=(7,7));x=np.array(host,order='F')
        # The slab is the larger view (61*3 = 183 numbers against 26*7 = 182), then the tile (61*2 = 122 against
        # 19*7 = 133), then the default, where both are the whole small basis.
        for size,width,tile in ((1500,3,26),(1100,2,19),(None,7,61)):
            with patch.object(ritz,'require_cupy',return_value=(shim,None)),patch.dict(os.environ):
                os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES',None)
                if size is not None:os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES']=str(size)
                slab,rotated=ritz._streaming_workspace(x)
                self.assertEqual((slab.shape,rotated.shape),((61,width),(tile,7)))
                self.assertTrue(slab.flags.f_contiguous and rotated.flags.f_contiguous)
                self.assertEqual(slab.ctypes.data,rotated.ctypes.data)
                # The bytes of that buffer, as a rule that counts them before the basis exists is given them.
                self.assertEqual(ritz.streaming_workspace_bytes(61,7),max(slab.nbytes,rotated.nbytes))
                # A shared buffer and buffers of their own give the same bits.
                separate=ritz._streamed_projection(operator,x);shared=ritz._streamed_projection(operator,x,slab)
                np.testing.assert_array_equal(shared,separate)
                np.testing.assert_allclose(np.tril(shared),np.tril(host.T@operator@host),rtol=1e-12,atol=1e-11)
                alone=x.copy(order='F');ritz._rotate_in_place(alone,c)
                moved=x.copy(order='F');ritz._rotate_in_place(moved,c,rotated)
                np.testing.assert_array_equal(moved,alone)
                np.testing.assert_allclose(moved,host@c,rtol=0,atol=1e-12)
                with self.assertRaises(ValueError):ritz._rotate_in_place(moved,c,slab[:,:1])
        # Sector sizes of C3469H804 and C4601H988.  With the 2 GiB of the default the tile is 1 MB and 20 MB larger
        # than the slab of the budget, and 0.76 and 0.63 GiB larger than the slab of 64 columns that is cut.  With
        # the former 4 GiB (PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES) it is 2 MB and 19 MB larger than the slab of
        # the budget, and 277 MB and 1.25 GiB larger than the slab of 192 and 128 columns that is cut.
        limit='PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES'
        with patch.dict(os.environ):
            for name in ('PARSEC_CUPY_STREAMING_RITZ_BYTES',limit):os.environ.pop(name,None)
            for named,cases in ((None,(((2604846,1842),(103,145730),64),((2873400,2430),(93,110467),64))),
                                (4<<30,(((2604846,1842),(206,291460),192),((2873400,2430),(186,220934),128)))):
                if named is not None:os.environ[limit]=str(named)
                for shape,expected,cut in cases:
                    sector=SimpleNamespace(shape=shape,nbytes=8*shape[0]*shape[1])
                    os.environ['PARSEC_CUPY_RITZ_GRAM_MULTIPLE']='1'
                    self.assertEqual(ritz._streaming_shape(sector),expected)
                    self.assertLess(shape[0]*expected[0],expected[1]*shape[1])
                    self.assertEqual(ritz.streaming_workspace_bytes(*shape),8*expected[1]*shape[1])
                    os.environ.pop('PARSEC_CUPY_RITZ_GRAM_MULTIPLE')
                    self.assertEqual(ritz._streaming_shape(sector),(cut,expected[1]))
                    # The rule that counts the buffer before the basis exists counts the slab of the budget at every
                    # multiple: here the tile, which the multiple leaves, is the larger view either way.
                    self.assertEqual(ritz.streaming_workspace_bytes(*shape),8*expected[1]*shape[1])

    def test_failed_audit_releases_the_shared_buffer_on_the_host(self):
        import importlib,weakref
        from types import SimpleNamespace
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        made=[]
        def empty(*args,**kwargs):
            array=np.empty(*args,**kwargs);made.append(weakref.ref(array));return array
        # NumPy stands in for CuPy as above; the one-array route asks ``empty`` for the shared buffer only.
        shim=SimpleNamespace(float64=np.float64,dtype=np.dtype,empty=empty,zeros=np.zeros,asarray=np.asarray,
                             asfortranarray=np.asfortranarray,stack=np.stack,asnumpy=np.asarray,
                             ndarray=lambda shape,dtype,memptr,order:np.ndarray(shape,dtype=dtype,buffer=memptr,order=order))
        rng=np.random.default_rng(71);operator=rng.normal(size=(61,61));operator+=operator.T
        host=rng.normal(size=(61,7));x=np.array(host,order='F');released=None
        def unstable(*args):
            self.assertIsNotNone(made[0]());raise ritz.GeneralizedRitzStabilityError('forced')
        # Whichever small solve the environment selects fails.
        with patch.object(ritz,'require_cupy',return_value=(shim,None)),patch.object(ritz,'solve_whitened_ritz',unstable), \
                patch.object(ritz,'solve_whitened_ritz_on_device',unstable), \
                patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='1',PARSEC_CUPY_STREAMING_RITZ_BYTES='1500',
                           PARSEC_CUPY_RITZ_SYRK='off'):
            # A caller's QR fallback runs inside its handler, where the traceback still refers to the frame of the
            # failed solve; assertRaises would clear that frame before anything could be checked.
            try:ritz.generalized_rayleigh_ritz(operator,x,consume_basis=True)
            except ritz.GeneralizedRitzStabilityError:released=[reference() is None for reference in made]
        self.assertEqual(released,[True])
        # The basis is rotated only after the audits: the fallback finds it as filtered.
        np.testing.assert_array_equal(x,host)

    def test_streaming_ritz_policy_is_on_off_or_decided_per_operator(self):
        from types import SimpleNamespace
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import streaming_ritz_requested
        marked,cleared=SimpleNamespace(low_memory_ritz=True),SimpleNamespace(low_memory_ritz=False)
        # Unset is auto: the mark of the operator's owner decides, and an operator nobody marked keeps two arrays.
        for value,expected in (('0',(False,False,False)),('1',(True,True,True)),('auto',(True,False,False)),
                               (None,(True,False,False))):
            with patch.dict(os.environ):
                os.environ.pop('PARSEC_CUPY_STREAMING_RITZ',None)
                if value is not None:os.environ['PARSEC_CUPY_STREAMING_RITZ']=value
                self.assertEqual((streaming_ritz_requested(marked),streaming_ritz_requested(cleared),
                                  streaming_ritz_requested()),expected)
        with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='sometimes'):
            with self.assertRaises(ValueError):streaming_ritz_requested()

    def test_every_unshared_sector_keeps_one_array_by_default(self):
        import importlib
        from types import SimpleNamespace
        symmetry=importlib.import_module('parsec_python.acceleration.Eigensolvers.symmetry')
        class Device:
            def __init__(self,index):pass
            def __enter__(self):return self
            def __exit__(self,*args):return False
        gib=1<<30
        cp=SimpleNamespace(cuda=SimpleNamespace(Device=Device,runtime=SimpleNamespace(memGetInfo=lambda:(80*gib,80*gib))))
        def marks(counts,**environment):
            # Four sectors of 2**20 rows, one per device; the third belongs to another rank and the first solve of
            # the fourth spread its basis over the devices of its group.
            solver=object.__new__(symmetry.CuPySymmetrySCFEigensolver)
            solver._memory_allocator_evaluated=False
            solver.device_ids=solver._sector_device_ids=(0,1,2,3)
            solver.decomposition=SimpleNamespace(sector_size=lambda representation:1<<20)
            solver._operators=[SimpleNamespace(),SimpleNamespace(),None,SimpleNamespace(_sector_device_group=object())]
            solver._solvers=[None if operator is None else object() for operator in solver._operators]
            with patch.object(symmetry,'require_cupy',return_value=(cp,None)),patch.dict(os.environ):
                for name in ('PARSEC_CUPY_STREAMING_RITZ','PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION'):
                    os.environ.pop(name,None)
                os.environ.update(PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR='pool',PARSEC_CUPY_SECTOR_STATE_STORAGE='device',
                                  **environment)
                # Before the marks are set, the report follows an explicit setting only.
                self.assertEqual(solver.low_memory_ritz_sectors,
                                 (0,1) if environment.get('PARSEC_CUPY_STREAMING_RITZ')=='1' else ())
                solver._configure_large_problem_allocator(counts)
                # The report reads the setting when it is asked for, so it is taken under that of the run.
                reported=solver.low_memory_ritz_sectors
            return [None if operator is None else operator.low_memory_ritz for operator in solver._operators],reported
        # 4 GiB and 32 GiB of basis: two arrays of the first would fit its device easily.
        counts=[512,4096,512,512]
        self.assertEqual(marks(counts),([True,True,None,True],(0,1)))
        self.assertEqual(marks(counts,PARSEC_CUPY_STREAMING_RITZ='auto'),([True,True,None,True],(0,1)))
        # An explicit setting is reported as it acts: no sector with 0, every unshared one with 1.
        self.assertEqual(marks(counts,PARSEC_CUPY_STREAMING_RITZ='0')[1],())
        self.assertEqual(marks(counts,PARSEC_CUPY_STREAMING_RITZ='1')[1],(0,1))
        # A fraction returns to two arrays where they stay within it: 2*4 + 3 GiB do, 2*32 + 3 GiB do not.
        self.assertEqual(marks(counts,PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION='0.8'),([False,True,None,False],(1,)))
        self.assertEqual(marks(counts,PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION='0.8',PARSEC_CUPY_STREAMING_RITZ='1')[1],
                         (0,1))
        for bad in ('-0.1','1.5'):
            with self.assertRaisesRegex(ValueError,'PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION'):
                marks(counts,PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION=bad)

    def test_gram_slab_count_sets_the_slab_width(self):
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import _gram_slab_width
        with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_MULTIPLE='1'):
            os.environ.pop('PARSEC_CUPY_RITZ_GRAM_SLABS',None)
            # 8 slabs by default, the last one narrower: 2430 = 7*304 + 302.  The count is an upper bound,
            # because the width is rounded up: 17 columns make 6 slabs of 3 and 5 columns make 5 of 1.
            self.assertEqual([_gram_slab_width(n) for n in (2430,1842,17,5,1)],[304,231,3,1,1])
            # Those are the widths of a multiple of 1.  By default a slab of 64 columns and more is cut down to
            # a whole multiple of 64, which makes more slabs: 2430 = 9*256 + 126.
            os.environ.pop('PARSEC_CUPY_RITZ_GRAM_MULTIPLE')
            self.assertEqual([_gram_slab_width(n) for n in (2430,1842,17,5,1)],[256,192,3,1,1])
        for slabs,width in (('1',17),('3',6),('17',1),('40',1)):
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS=slabs):
                self.assertEqual(_gram_slab_width(17),width)
        for value in ('0','-3','many'):
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS=value):
                with self.assertRaises(ValueError):_gram_slab_width(17)

    def test_gram_slabs_are_cut_at_whole_multiples_of_64_columns(self):
        import importlib
        from types import SimpleNamespace
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        name='PARSEC_CUPY_RITZ_GRAM_MULTIPLE'
        # Rows and columns of a sector of 3,480, 5,264, 7,120, 10,456, 14,680, 19,392, 23,768, 29,576 and 39,368
        # electrons, the nine clusters of the measured series.
        sectors=((677542,442),(1031322,665),(1194720,897),(1924792,1314),(2604846,1842),(2873400,2431),
                 (3786832,2978),(4130510,3704),(5660200,4928))
        def widths():
            shapes=[ritz._streaming_shape(SimpleNamespace(shape=shape,nbytes=8*shape[0]*shape[1])) for shape in sectors]
            return ([ritz._gram_slab_width(columns) for _rows,columns in sectors],[slab for slab,_tile in shapes],
                    [tile for _slab,tile in shapes])
        with patch.dict(os.environ):
            for unset in (name,'PARSEC_CUPY_RITZ_GRAM_SLABS','PARSEC_CUPY_STREAMING_RITZ_BYTES'):
                os.environ.pop(unset,None)
            # The buffer of one array as it was when the multiple came: at most 4 GiB, where it takes 2 since.
            os.environ['PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES']=str(4<<30)
            self.assertEqual(ritz.gram_multiple(),64)
            overlap,projection,tiles=widths()
            # The slabs of the overlap, an eighth of the columns, and those of the projection, the columns of
            # the streaming budget: whole multiples of 64 from 64 columns on, and what they were below.
            self.assertEqual(overlap,[56,64,64,128,192,256,320,448,576])
            self.assertEqual(projection,[192,128,64,128,192,128,128,128,64])
            os.environ[name]='1'
            former=widths()
            self.assertEqual(former[0],[56,84,113,165,231,304,373,463,616])
            self.assertEqual(former[1],[198,130,112,164,206,186,141,129,94])
            # A slab only narrows, to the multiple below it, and the rotation tile is what it was.
            for cut,before in zip(overlap+projection,former[0]+former[1]):
                self.assertEqual(cut,before-before%64 if before>=64 else before)
            self.assertEqual(tiles,former[2])
            # The widths that ran on A100 nodes with the switches set by hand, 192 columns of 4,001,043,456
            # bytes and 128 of 1,056,073,728, are multiples already: the budget is the slab either way.
            for value in ('1','64'):
                os.environ[name]=value
                for rows,columns,budget,width in ((2604846,1842,4001043456,192),(1031322,665,1056073728,128)):
                    with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ_BYTES=str(budget)):
                        sector=SimpleNamespace(shape=(rows,columns),nbytes=8*rows*columns)
                        self.assertEqual(ritz._streaming_shape(sector)[0],width)
            # Any multiple can be named.  One slab is the full product and a basis that fits its budget is one
            # slab, whatever the multiple.
            for value,expected in (('32',(224,192)),('128',(128,128)),('100',(200,200)),('1000',(231,206))):
                os.environ[name]=value
                sector=SimpleNamespace(shape=(2604846,1842),nbytes=8*2604846*1842)
                self.assertEqual((ritz._gram_slab_width(1842),ritz._streaming_shape(sector)[0]),expected)
            os.environ[name]='64'
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS='1'):
                self.assertEqual([ritz._gram_slab_width(columns) for columns in (100,1842)],[100,1842])
            self.assertEqual(ritz._streaming_shape(SimpleNamespace(shape=(164882,300),nbytes=8*164882*300)),(300,164882))
            self.assertEqual([ritz._whole_multiple(width) for width in (1,63,64,65,127,128,206)],[1,63,64,64,64,128,192])
            self.assertEqual([ritz._whole_multiple(width,48) for width in (47,48,100)],[47,48,96])
            for value in ('0','-64','many','','6.4'):
                os.environ[name]=value
                for read in (ritz.gram_multiple,lambda:ritz._gram_slab_width(1842)):
                    with self.assertRaisesRegex(ValueError,name):read()
        # The slab loops need no device: NumPy stands in for CuPy.  17 columns in at most 3 slabs are 6 wide,
        # and 4 with a multiple of 4: slabs at 0, 4, 8, 12 and the last column.
        rng=np.random.default_rng(73);operator=rng.normal(size=(61,61));operator+=operator.T
        x,y=(np.asfortranarray(rng.normal(size=(61,17))) for _ in range(2));whole=x.T@y
        shim=SimpleNamespace(float64=np.float64,dtype=np.dtype,empty=np.empty,zeros=np.zeros)
        def former(left,right,width):
            # Slab by slab as the loops form it; ``right`` gives the columns of a slab in a buffer of its own.
            product=np.zeros((17,17),order='F')
            for start in range(0,17,width):
                slab=np.empty((61,min(17,start+width)-start),order='F');slab[...]=right(start,start+width)
                product[start:,start:start+width]=left[:,start:].T@slab
            return product
        # Each slab is one product here: the slabs are 4 and 6 columns wide, and an even width below 64 would
        # be multiplied in two pieces, which test_tall_product covers.
        with patch.object(ritz,'require_cupy',return_value=(shim,None)), \
                patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS='3',PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*61*6),
                           PARSEC_CUPY_TALL_PRODUCT_SPLIT='0'):
            for value,width in (('4',4),('1',6),('64',6),(None,6)):
                os.environ.pop(name,None)
                if value is not None:os.environ[name]=value
                lower=ritz._lower_triangle_product(x,y)
                np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-13,atol=1e-13)
                # The slabs of the width that was cut, to the last bit, and nothing above them.
                np.testing.assert_array_equal(lower,former(x,lambda first,last:y[:,first:last],width))
                for start in range(width,17,width):self.assertFalse(lower[:start,start:start+width].any())
                projected=ritz._streamed_projection(operator,x)
                np.testing.assert_array_equal(projected,former(x,lambda first,last:operator@x[:,first:last],width))
                self.assertEqual(ritz._streaming_shape(x),(width,min(61,6*61//17)))

    def test_the_buffer_of_one_array_takes_2_gib_at_most_unless_its_limit_or_a_budget_is_named(self):
        import importlib
        from types import SimpleNamespace
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        gib=1<<30;limit,budget='PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES','PARSEC_CUPY_STREAMING_RITZ_BYTES'
        def former(rows,columns):
            # The slab and the tile as the route cut them: an eighth of the basis within 1 and 4 GiB.
            held=min(max(rows*columns,gib),4*gib)
            return max(1,min(columns,held//(8*rows))),max(1,min(rows,held//(8*columns)))
        def cut(rows,columns):return ritz._streaming_cut(rows,columns,8*rows*columns)
        # Rows from a few hundred to 20 million, among them those of the measured sectors and the two row counts
        # at which 2 and 4 GiB hold 64 columns; columns from one to 20,000.
        many=(257,65_537,677_542,1_194_720,1_924_792,2_604_846,2_873_400,4_194_304,4_194_305,5_660_200,8_388_608,
              8_388_609,20_000_000)
        shapes=[(rows,columns) for rows in many for columns in (1,17,63,64,65,442,897,1024,1314,1842,2431,4928,20_000)]
        with patch.dict(os.environ):
            for name in (limit,budget,'PARSEC_CUPY_RITZ_GRAM_MULTIPLE'):os.environ.pop(name,None)
            # The limit follows the rows of a sector and nothing else of it: 2 GiB while they hold 64 columns,
            # then those 64 columns, and from the rows of which 4 GiB hold no more the 4 GiB that it was.
            self.assertEqual([ritz.one_array_buffer_limit(rows) for rows in (0,257,4_194_304,4_194_305,5_660_200,8_388_608,
                                                                             8_388_609,10**8)],
                             [2*gib,2*gib,2*gib,512*4_194_305,512*5_660_200,4*gib,4*gib,4*gib])
            for rows,columns in shapes:
                basis=8*rows*columns;width,tile=cut(rows,columns);before=former(rows,columns)
                held=min(max(basis//8,gib),max(2*gib,min(512*rows,4*gib)))
                # An eighth of the basis within 1 GiB and the limit: as many columns and rows as that holds.
                self.assertEqual((width,tile),(max(1,min(columns,held//(8*rows))),max(1,min(rows,held//(8*columns)))))
                # Never more than the buffer was, the buffer that was up to 16 GiB of basis, and no slab below
                # 64 columns that had them.
                self.assertLessEqual(width,before[0]);self.assertLessEqual(tile,before[1])
                if basis<=16*gib:self.assertEqual((width,tile),before)
                if before[0]>=64:self.assertGreaterEqual(width,64)
                # What a rule counts for it before the basis exists.
                self.assertEqual(ritz.streaming_workspace_bytes(rows,columns),8*max(rows*width,tile*columns))
                self.assertLessEqual(ritz.streaming_workspace_bytes(rows,columns),max(held,8*rows,8*columns))
            # A slab never narrows as columns are added, and the bytes of the budget never fall: a later step of
            # a sector, which holds as many columns as its first solve or fewer, takes no larger buffer.
            for rows in (65_537,1_924_792,2_873_400,5_660_200):
                widths=[cut(rows,columns)[0] for columns in range(1,6001,13)]
                budgets=[ritz._streaming_bytes(8*rows*columns,ritz.one_array_buffer_limit(rows)) for columns in range(1,6001,13)]
                self.assertEqual((widths,budgets),(sorted(widths),sorted(budgets)))
            # The sectors of 10,456, 14,680 and 19,392 electrons, 18.8, 35.7 and 52.0 GiB: the slab of the first
            # stays 128 columns under a lower tile, those of the others are 64 columns where they were 192 and 128.
            measured=((1_924_792,1314),(2_604_846,1842),(2_873_400,2431))
            shape=lambda rows,columns:ritz._streaming_shape(SimpleNamespace(shape=(rows,columns),nbytes=8*rows*columns))
            self.assertEqual([shape(*sector) for sector in measured],[(128,204288),(64,145730),(64,110421)])
            self.assertEqual([round(ritz.streaming_workspace_bytes(*sector)/gib,3) for sector in measured],[2.,2.,2.])
            # The former value of the limit gives every shape the slab and the tile that it had, a budget that is
            # set is the buffer whatever the limit says, and the steps of a shared basis read no limit.
            os.environ[limit]=str(4*gib)
            self.assertEqual([ritz.one_array_buffer_limit(rows) for rows in (0,257,5_660_200,10**8)],[4*gib]*4)
            for rows,columns in shapes:self.assertEqual(cut(rows,columns),former(rows,columns))
            self.assertEqual([shape(*sector) for sector in measured],[(128,240599),(192,291460),(128,220843)])
            self.assertEqual([round(ritz.streaming_workspace_bytes(*sector)/gib,3) for sector in measured],[2.355,4.,4.])
            for named in (str(gib),str(4*gib),None):
                os.environ.pop(limit,None)
                if named is not None:os.environ[limit]=named
                with patch.dict(os.environ,{budget:str(3*gib)}):
                    self.assertEqual([cut(*sector) for sector in measured],
                                     [(3*gib//(8*rows),3*gib//(8*columns)) for rows,columns in measured])
                self.assertEqual([ritz._streaming_bytes(n<<30) for n in (0,16,64)],[1<<30,2<<30,4<<30])
            # A limit below 1 GiB is the buffer of every basis; the 4 GiB of the budget are not the way back.
            os.environ[limit]=str(8*1000*70)
            self.assertEqual(cut(1000,600),(70,116))
            os.environ.pop(limit)
            with patch.dict(os.environ,{budget:str(4*gib)}):
                self.assertNotEqual(cut(*measured[0]),former(*measured[0]))
            for bad in ('0','-1','2GiB','','2.5'):
                os.environ[limit]=bad
                for read in (lambda:ritz.one_array_buffer_limit(1000),lambda:cut(1000,600),
                             lambda:ritz.streaming_workspace_bytes(1000,600)):
                    with self.assertRaisesRegex(ValueError,limit):read()

    def test_the_rotation_tile_of_one_array_takes_the_bytes_of_the_budget(self):
        import importlib
        from types import SimpleNamespace
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        budget='PARSEC_CUPY_STREAMING_RITZ_BYTES'
        shape=lambda rows,columns:ritz._streaming_shape(SimpleNamespace(shape=(rows,columns),nbytes=8*rows*columns))
        # Sectors of 3,480 to 19,392 electrons and one of many rows; the buffer of each in bytes.
        sectors=((677_542,442),(1_031_322,665),(1_194_720,897),(1_924_792,1314),(2_604_846,1842),(2_873_400,2431),
                 (5_660_200,1200))
        taken=lambda rows,columns:8*max(rows*shape(rows,columns)[0],shape(rows,columns)[1]*columns)
        with patch.dict(os.environ):
            for unset in (budget,'PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES','PARSEC_CUPY_RITZ_GRAM_MULTIPLE'):
                os.environ.pop(unset,None)
            default=[shape(*sector) for sector in sectors];held=[taken(*sector) for sector in sectors]
            counted=[ritz.streaming_workspace_bytes(*sector) for sector in sectors]
            # The tile of the budget: the buffer is the budget, to less than a row, whatever the slab is cut to.
            self.assertEqual(default,[(192,303660),(128,201831),(64,149629),(128,204288),(64,145730),(64,110421),(64,301877)])
            self.assertEqual(held,counted)
            self.assertEqual([round(count/2**30,2) for count in held],[1.,1.,1.,2.,2.,2.,2.7])
            # The count is the slab of the budget, which no multiple moves.
            os.environ['PARSEC_CUPY_RITZ_GRAM_MULTIPLE']='1'
            self.assertEqual([ritz.streaming_workspace_bytes(*sector) for sector in sectors],counted)
            os.environ.pop('PARSEC_CUPY_RITZ_GRAM_MULTIPLE')
            # A basis that its budget holds is one slab and one tile, and a budget whose slab the multiple cuts
            # keeps the tile of its bytes: 70 columns of 1,000 rows are cut to 64, 60 are not.
            self.assertEqual(shape(164_882,300),(300,164_882))
            for columns,cut in ((70,(64,116)),(60,(60,100))):
                with patch.dict(os.environ,{budget:str(8*1000*columns)}):
                    self.assertEqual(shape(1000,600),cut)
        # The two views of one buffer and a rotation in the tile, NumPy standing in for CuPy: 61 rows by 7 columns
        # under 1,100 bytes are slabs of 2 columns with a tile of 19 rows.
        shim=SimpleNamespace(float64=np.float64,dtype=np.dtype,empty=np.empty,zeros=np.zeros,
                             asfortranarray=np.asfortranarray,ascontiguousarray=np.ascontiguousarray,
                             ndarray=lambda shape,dtype,memptr,order:np.ndarray(shape,dtype=dtype,buffer=memptr,order=order))
        rng=np.random.default_rng(97);host=rng.normal(size=(61,7));c=rng.normal(size=(7,7));x=np.array(host,order='F')
        with patch.object(ritz,'require_cupy',return_value=(shim,None)),patch.dict(os.environ,{budget:'1100'}):
            slab,rotated=ritz._streaming_workspace(x)
            self.assertEqual((slab.shape,rotated.shape,slab.ctypes.data),((61,2),(19,7),rotated.ctypes.data))
            self.assertEqual(max(slab.size,rotated.size),133)
            self.assertEqual(ritz.streaming_workspace_bytes(61,7),8*133)
            ritz._rotate_in_place(x,c,rotated)
        np.testing.assert_allclose(x,host@c,rtol=0,atol=1e-12)

    def test_lower_triangle_slabs_match_the_dense_product_on_the_host(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        rng=np.random.default_rng(57)
        x,y=(np.asfortranarray(rng.normal(size=(61,17))) for _ in range(2));whole=x.T@y
        # The slab loop needs no device: NumPy stands in for CuPy.
        with patch.object(ritz,'require_cupy',return_value=(np,None)):
            for slabs,width in (('1',17),('3',6),('8',3),('17',1)):
                with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS=slabs):
                    lower=ritz._lower_triangle_product(x,y)
                np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-13,atol=1e-13)
                # Nothing is computed above the diagonal slabs; one slab is the full product.
                for start in range(width,17,width):self.assertFalse(lower[:start,start:start+width].any())
                self.assertEqual(bool(np.triu(lower,6).any()),slabs=='1')
            # 8 slabs of 2 columns out of 16 multiply 144 of 256 column pairs, 56%.
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS='8'):
                self.assertEqual(np.count_nonzero(ritz._lower_triangle_product(np.ones((3,16)),np.ones((3,16)))),144)

    def test_a_caller_names_the_slab_width_of_a_product_and_the_count_is_then_not_read(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        rng=np.random.default_rng(79)
        x,y=(np.asfortranarray(rng.normal(size=(61,17))) for _ in range(2));whole=x.T@y
        # The overlap of a row block of a shared basis is cut by a width that its step names.  NumPy stands in
        # for CuPy; the count of the environment gives slabs of 6 columns.
        with patch.object(ritz,'require_cupy',return_value=(np,None)), \
                patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS='3',PARSEC_CUPY_RITZ_SYRK='auto',PARSEC_CUPY_RITZ_GRAM_MULTIPLE='1'), \
                patch.object(ritz,'_gram_slab_width',wraps=ritz._gram_slab_width) as asked:
            counted=ritz._lower_triangle_product(x,y);asked.assert_called_once_with(17)
            for width in (1,4,6,16,17,40):
                lower=ritz._lower_triangle_product(x,y,width)
                np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-13,atol=1e-13)
                for start in range(width,17,width):self.assertFalse(lower[:start,start:start+width].any())
                # The slabs of the count where the width is theirs, and the full product where one slab holds all.
                if width==6:np.testing.assert_array_equal(lower,counted)
                self.assertEqual(bool(lower[0,16]),width>=17)
            np.testing.assert_array_equal(ritz._symmetric_overlap(x,4),ritz._lower_triangle_product(x,x,4))
            # The count was read for the one product that was given no width.
            asked.assert_called_once_with(17)
            # Without a width the overlap is the call that it was, and the full product reads no width.
            np.testing.assert_array_equal(ritz._symmetric_overlap(x),ritz._lower_triangle_product(x,x,6))
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='off'):
                np.testing.assert_array_equal(ritz._symmetric_overlap(x,4),x.T@x)
        # Which of the three forms the overlap, for the step that reports the width of its slabs.
        for value,policy in (('auto','slabs'),('slabs','slabs'),('on','syrk'),('1','syrk'),('true','syrk'),
                             ('off','full'),('0','full'),('false','full')):
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK=value):self.assertEqual(ritz._overlap_policy(),policy)
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_RITZ_SYRK',None);self.assertEqual(ritz._overlap_policy(),'slabs')
        with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='sometimes'):
            with self.assertRaises(ValueError):ritz._overlap_policy()

    def test_overlap_policy_selects_slabs_or_the_full_product_on_the_host(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        x=np.asfortranarray(np.random.default_rng(59).normal(size=(61,17)));whole=x.T@x
        # DSYRK itself needs cuBLAS; the other policies run with NumPy standing in for CuPy.
        with patch.object(ritz,'require_cupy',return_value=(np,None)):
            # The slabs are the default: unset and auto select them as the explicit name does.
            for policy in (None,'auto','slabs'):
                with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS='3'):
                    os.environ.pop('PARSEC_CUPY_RITZ_SYRK',None)
                    if policy is not None:os.environ['PARSEC_CUPY_RITZ_SYRK']=policy
                    lower=ritz._symmetric_overlap(x)
                np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-13,atol=1e-13)
                self.assertFalse(np.triu(lower,6).any())
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='off',PARSEC_CUPY_RITZ_GRAM_SLABS='3'):
                full=ritz._symmetric_overlap(x)
            np.testing.assert_allclose(full,whole,rtol=1e-13,atol=1e-13)
            # DSYRK is taken only when it is named, and nothing stands in for it then.
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='on'),patch.object(ritz,'_lower_triangle_product') as slabs:
                with self.assertRaises(Exception):ritz._symmetric_overlap(x)
            slabs.assert_not_called()
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='sometimes'):
                with self.assertRaises(ValueError):ritz._symmetric_overlap(x)

    def test_no_lost_or_repeated_blocks_for_all_counts(self):
        for n in range(1,24):
            blocks=tuple(FilterBlock(i*6,(i+1)*6,9 if i<n//2 else 15) for i in range(n))
            for devices in (1,2,3,4):
                partitions=block_partitions(blocks,devices)
                self.assertEqual([i for a,b in partitions for i in range(a,b)],list(range(n)))
                self.assertTrue(all(a<b for a,b in partitions))

    def test_column_copy_flag_and_rotation_on_the_host(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        for value,expected in (('0',False),('off',False),('1',True),('on',True)):
            with patch.dict(os.environ,PARSEC_CUPY_ROTATE_COLUMN_COPY=value):
                self.assertEqual(ritz._column_copy_requested(),expected)
        # Along columns unless switched off.
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_ROTATE_COLUMN_COPY',None);self.assertTrue(ritz._column_copy_requested())
        with patch.dict(os.environ,PARSEC_CUPY_ROTATE_COLUMN_COPY='columns'):
            with self.assertRaises(ValueError):ritz._column_copy_requested()
        rng=np.random.default_rng(61);host=rng.normal(size=(61,7));c=rng.normal(size=(7,7));moved={}
        # NumPy stands in for CuPy, so tiles take the generic product; 13 rows per tile leave a remainder of 9.
        with patch.object(ritz,'require_cupy',return_value=(np,None)):
            for flag in ('0','1'):
                with patch.dict(os.environ,PARSEC_CUPY_ROTATE_COLUMN_COPY=flag,PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*7*13)):
                    moved[flag]=np.array(host,order='F');ritz._rotate_in_place(moved[flag],c)
        np.testing.assert_array_equal(moved['1'],moved['0'])
        np.testing.assert_allclose(moved['1'],host@c,rtol=0,atol=1e-12)

@unittest.skipUnless(cupy_available(),'CUDA required')
class NextDeviceTests(unittest.TestCase):
    def setUp(self):
        self.cp,_=require_cupy();self.cp.cuda.Device(0).use()
        n=257
        self.op=CuPyHamiltonian(sp.diags((-np.ones(n-1),2.2*np.ones(n),-np.ones(n-1)),(-1,0,1)),
                               np.linspace(-.1,.1,n),
                               (sp.csr_matrix(np.random.default_rng(8).normal(size=(n,2))*.002),np.array([1.,-1.])),
                               retain_generic_laplacian=False)

    def tearDown(self):
        # Solvers and filter graphs that a test left in a reference cycle are destroyed here, between the tests,
        # and not by a collection inside the next one.
        gc.collect()

    def test_distributed_filter_matches_single_device_with_projectors_and_partial_block(self):
        cp=self.cp
        if cp.cuda.runtime.getDeviceCount()<2:self.skipTest('multiple GPUs required')
        x=cp.array(np.random.default_rng(11).normal(size=(257,23)),order='F')
        with patch.dict(os.environ,PARSEC_CUPY_FILTER_GRAPHS='1',PARSEC_CUPY_MIXED_FILTER='off',PARSEC_CUPY_FILTER_COLUMN_MAJOR='1'):
            for reset in (False,True):
                with patch.dict(os.environ,PARSEC_CUPY_DISTRIBUTED_FILTER='0'):
                    expected=subspace_filter(self.op,x,9,3,1.2,6.,reset_recurrence_per_block=reset)
                for count in range(2,min(4,cp.cuda.runtime.getDeviceCount())+1):
                    if hasattr(self.op,'_distributed_filter'):del self.op._distributed_filter
                    with patch.dict(os.environ,PARSEC_CUPY_DISTRIBUTED_FILTER='1',PARSEC_CUPY_DEVICES=','.join(map(str,range(count)))):
                        actual=subspace_filter(self.op,x,9,3,1.2,6.,reset_recurrence_per_block=reset)
                    np.testing.assert_array_equal(cp.asnumpy(actual),cp.asnumpy(expected))
        del self.op._distributed_filter

    def test_assigned_device_groups_override_global_devices_and_preserve_filter(self):
        import importlib
        from threading import Lock
        distributed = importlib.import_module(
            'parsec_python.acceleration.Eigensolvers.distributed_filter')
        cp = self.cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 2:
            self.skipTest('requires at least two allocated GPUs')
        groups = ((0, 1), (2, 3)) if device_count >= 4 else ((0,), (1,))
        host_x = np.random.default_rng(31).normal(size=(257, 23))
        kinetic = sp.diags((-np.ones(256), 2.2*np.ones(257), -np.ones(256)), (-1, 0, 1))
        projectors = sp.csr_matrix(np.random.default_rng(8).normal(size=(257, 2))*.002)
        potential = np.linspace(-.1, .1, 257)
        original_apply = distributed.BlockFilterGraphs.apply
        original_operator = distributed.CuPyHamiltonian
        original_pool = distributed._pool
        lock = Lock()
        with patch.dict(os.environ, PARSEC_CUPY_FILTER_GRAPHS='1',
                        PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1',
                        PARSEC_CUPY_DEVICES=','.join(map(str, range(device_count)))):
            for group in groups:
                with self.subTest(group=group), cp.cuda.Device(group[0]):
                    operator = CuPyHamiltonian(kinetic, potential,
                        (projectors, np.array([1., -1.])), retain_generic_laplacian=False)
                    operator.distributed_filter_devices = group
                    x = cp.array(host_x, order='F')
                    executed, constructed, scheduled = set(), set(), set()

                    def record_graph(graph, *args, **kwargs):
                        device = int(cp.cuda.Device().id)
                        self.assertIn(device, group, 'filter ran outside its assigned group')
                        with lock:
                            executed.add(device)
                        return original_apply(graph, *args, **kwargs)

                    def record_operator(*args, **kwargs):
                        device = int(cp.cuda.Device().id)
                        self.assertIn(device, group, 'replica allocated outside its assigned group')
                        constructed.add(device)
                        return original_operator(*args, **kwargs)

                    def record_pool(device):
                        self.assertIn(device, group, 'work submitted outside its assigned group')
                        scheduled.add(device)
                        return original_pool(device)

                    # Reuse persistent workers across changed SCF potentials,
                    # while exercising carried sigma and explicit reset paths.
                    for shift in (0., .031):
                        operator.effective_potential.set(potential+shift)
                        for reset in (False, True):
                            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_FILTER='0'):
                                expected = cp.asnumpy(subspace_filter(operator, x, 9, 3, 1.2, 6.,
                                    reset_recurrence_per_block=reset))
                            with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_FILTER='1'), \
                                 patch.object(distributed.BlockFilterGraphs, 'apply', record_graph), \
                                 patch.object(distributed, 'CuPyHamiltonian', record_operator), \
                                 patch.object(distributed, '_pool', record_pool):
                                actual = subspace_filter(operator, x, 9, 3, 1.2, 6.,
                                                         reset_recurrence_per_block=reset)
                            self.assertEqual(int(actual.device.id), group[0])
                            np.testing.assert_array_equal(cp.asnumpy(actual), expected)
                            np.testing.assert_array_equal(cp.asnumpy(x), host_x)
                    self.assertEqual(executed, set(group))
                    if len(group) > 1:
                        worker = operator._distributed_filter
                        self.assertEqual(worker.devices, group)
                        self.assertEqual(set(worker.streams), set(group))
                        self.assertEqual(set(worker.graphs), set(group))
                        self.assertEqual(set(worker.replicas), set(group)-{group[0]})
                        self.assertEqual(constructed, set(group)-{group[0]})
                        self.assertEqual(scheduled, set(group))
                    else:
                        self.assertFalse(hasattr(operator, '_distributed_filter'))
                        self.assertEqual(constructed, set())
                        self.assertEqual(scheduled, set())

    def test_consumed_input_matches_preserved_input_for_two_updates(self):
        cp=self.cp
        x=cp.array(np.linalg.qr(np.random.default_rng(17).normal(size=(257,17)))[0],order='F')
        original=x.copy(order='F')
        state=DeviceSubspaceState(257,17,cp.linspace(.1,1.5,17),x)
        owned=replace(state,vectors=x.copy(order='F'))
        with patch.dict(os.environ,PARSEC_CUPY_GENERALIZED_RITZ='on',PARSEC_CUPY_RITZ_ROTATION='reuse',PARSEC_CUPY_DISTRIBUTED_FILTER='0',PARSEC_CUPY_MIXED_FILTER='off'):
            settings=SubspaceSettings(polynomial_degree=3,degree_delta=1)
            for _ in range(2):
                a=run_subspace_filter(self.op,state,settings=settings,compute_residuals=False)
                b=run_subspace_filter(self.op,owned,settings=settings,compute_residuals=False,consume_state=True)
                np.testing.assert_allclose(cp.asnumpy(a.eigenvalues),cp.asnumpy(b.eigenvalues),rtol=0,atol=2e-13)
                np.testing.assert_allclose(cp.asnumpy(a.vectors),cp.asnumpy(b.vectors),rtol=0,atol=2e-12)
                self.assertIsNone(b.state.ritz_workspace)
                state,owned=a.state,b.state
        np.testing.assert_array_equal(cp.asnumpy(x),cp.asnumpy(original))

    def test_chebff_generalized_cycles_match_the_orthonormal_route(self):
        import importlib
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        # The package also exports functions named like these submodules.
        chebff_module=importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff')
        ritz_module=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        run_chebff=chebff_module.run_chebff;original=ritz_module.generalized_rayleigh_ritz
        cp=self.cp;settings=ChebFFSettings(polynomial_degree=12,filter_cycles=4)
        common=dict(PARSEC_CUPY_GENERALIZED_RITZ='on',PARSEC_CUPY_MIXED_FILTER='off',PARSEC_CUPY_DISTRIBUTED_FILTER='0')
        with patch.dict(os.environ,PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='0',**common):
            expected=run_chebff(self.op,19,settings=settings)
        with patch.dict(os.environ,PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1',**common),                 patch.object(chebff_module,'generalized_rayleigh_ritz',wraps=original) as direct:
            actual=run_chebff(self.op,19,settings=settings)
        self.assertEqual(direct.call_count,4)
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues),cp.asnumpy(expected.eigenvalues),rtol=0,atol=5e-11)
        # Both routes return an orthonormal Ritz basis of the same subspace.
        a,b=cp.asnumpy(actual.vectors),cp.asnumpy(expected.vectors)
        np.testing.assert_allclose(a.T@a,np.eye(19),atol=1e-9)
        np.testing.assert_allclose(np.linalg.svd(a.T@b,compute_uv=False),np.ones(19),atol=5e-9)
        self.assertEqual([c.number for c in actual.cycles],[1,2,3,4])
        self.assertIsNone(actual.last_rayleigh_ritz.workspace)
        # An unsafe overlap falls back to the orthonormal route in that cycle only.
        calls=[]
        def unstable_first(*args,**kwargs):
            calls.append(1)
            if len(calls)==1:raise ritz_module.GeneralizedRitzStabilityError('forced')
            return original(*args,**kwargs)
        with patch.dict(os.environ,PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1',**common),                 patch.object(chebff_module,'generalized_rayleigh_ritz',unstable_first):
            recovered=run_chebff(self.op,19,settings=settings)
        self.assertEqual(len(calls),4)
        np.testing.assert_allclose(cp.asnumpy(recovered.eigenvalues),cp.asnumpy(expected.eigenvalues),rtol=0,atol=5e-11)

    def test_in_place_filter_matches_out_of_place_on_one_and_several_devices(self):
        cp=self.cp;x=cp.array(np.random.default_rng(21).normal(size=(257,23)),order='F')
        counts=[1]+list(range(2,min(4,cp.cuda.runtime.getDeviceCount())+1))
        with patch.dict(os.environ,PARSEC_CUPY_FILTER_GRAPHS='1',PARSEC_CUPY_MIXED_FILTER='off',PARSEC_CUPY_FILTER_COLUMN_MAJOR='1'):
            with patch.dict(os.environ,PARSEC_CUPY_DISTRIBUTED_FILTER='0'):
                expected=cp.asnumpy(subspace_filter(self.op,x,9,3,1.2,6.))
            for count in counts:
                if hasattr(self.op,'_distributed_filter'):del self.op._distributed_filter
                owned=x.copy(order='F');pointer=owned.data.ptr
                with patch.dict(os.environ,PARSEC_CUPY_DISTRIBUTED_FILTER='0' if count==1 else '1',PARSEC_CUPY_DEVICES=','.join(map(str,range(count)))):
                    actual=subspace_filter(self.op,owned,9,3,1.2,6.,out=owned)
                self.assertIs(actual,owned);self.assertEqual(actual.data.ptr,pointer)
                np.testing.assert_array_equal(cp.asnumpy(actual),expected)
        if hasattr(self.op,'_distributed_filter'):del self.op._distributed_filter

    def test_streaming_ritz_keeps_one_tall_array_and_matches_the_two_buffer_route(self):
        cp=self.cp
        x=cp.array(np.linalg.qr(np.random.default_rng(19).normal(size=(257,17)))[0],order='F')
        reference=DeviceSubspaceState(257,17,cp.linspace(.1,1.5,17),x.copy(order='F'))
        owned=replace(reference,vectors=x.copy(order='F'));pointer=owned.vectors.data.ptr
        common=dict(PARSEC_CUPY_GENERALIZED_RITZ='on',PARSEC_CUPY_RITZ_ROTATION='reuse',PARSEC_CUPY_DISTRIBUTED_FILTER='0',
                    PARSEC_CUPY_MIXED_FILTER='off',PARSEC_CUPY_FILTER_GRAPHS='1',PARSEC_CUPY_FILTER_COLUMN_MAJOR='1')
        settings=SubspaceSettings(polynomial_degree=3,degree_delta=1)
        # 5 columns per slab and 75 rows per tile exercise several slabs and tiles.
        streaming=dict(PARSEC_CUPY_STREAMING_RITZ='1',PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*257*5))
        for _ in range(2):
            with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='0',**common):
                a=run_subspace_filter(self.op,reference,settings=settings,compute_residuals=False,consume_state=True)
            with patch.dict(os.environ,**common,**streaming):
                b=run_subspace_filter(self.op,owned,settings=settings,compute_residuals=False,consume_state=True)
            self.assertEqual(b.rayleigh_ritz.algorithm,'streaming_generalized_cholesky_rayleigh_ritz')
            # Filtered, projected and rotated in the caller's single buffer.
            self.assertEqual(b.vectors.data.ptr,pointer);self.assertIsNone(b.state.ritz_workspace)
            np.testing.assert_allclose(cp.asnumpy(a.eigenvalues),cp.asnumpy(b.eigenvalues),rtol=0,atol=5e-12)
            u,v=cp.asnumpy(a.vectors),cp.asnumpy(b.vectors)
            np.testing.assert_allclose(v.T@v,np.eye(17),atol=1e-9)
            np.testing.assert_allclose(np.linalg.svd(u.T@v,compute_uv=False),np.ones(17),atol=5e-9)
            reference,owned=a.state,b.state

    def test_streaming_ritz_shares_one_buffer_between_slab_and_tile(self):
        import importlib
        from scipy.linalg import eigh
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;rng=np.random.default_rng(41);host=rng.normal(size=(257,17));factors=rng.normal(size=(17,17))
        x=cp.array(host,order='F');c=cp.asarray(factors);whole=host.T@cp.asnumpy(self.op@x)
        # The slab is the larger view (257*5 = 1285 numbers against 75*17 = 1275), then the tile (257*3 = 771
        # against 56*17 = 952, as in production), then the default, where both are the whole small basis.
        for size,width,tile in ((8*257*5,5,75),(7700,3,56),(None,17,257)):
            with patch.dict(os.environ):
                os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES',None)
                if size is not None:os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES']=str(size)
                slab,rotated=ritz._streaming_workspace(x)
                self.assertEqual((slab.shape,rotated.shape),((257,width),(tile,17)))
                self.assertTrue(slab.flags.f_contiguous and rotated.flags.f_contiguous)
                self.assertEqual(slab.data.ptr,rotated.data.ptr)
                # A shared buffer and buffers of their own give the same bits.
                separate=cp.asnumpy(ritz._streamed_projection(self.op,x))
                shared=cp.asnumpy(ritz._streamed_projection(self.op,x,slab))
                np.testing.assert_array_equal(shared,separate)
                np.testing.assert_allclose(np.tril(shared),np.tril(whole),rtol=1e-12,atol=1e-10)
                alone=x.copy(order='F');ritz._rotate_in_place(alone,c)
                moved=x.copy(order='F');pointer=moved.data.ptr;ritz._rotate_in_place(moved,c,rotated)
                self.assertEqual(moved.data.ptr,pointer)
                np.testing.assert_array_equal(cp.asnumpy(moved),cp.asnumpy(alone))
                np.testing.assert_allclose(cp.asnumpy(moved),host@factors,rtol=0,atol=1e-12)
        # A whole one-array solve hands the two views of one buffer to its two steps.
        expected=eigh(np.tril(whole)+np.tril(whole,-1).T,host.T@host,eigvals_only=True)
        with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='1',PARSEC_CUPY_STREAMING_RITZ_BYTES='7700'), \
                patch.object(ritz,'_streamed_projection',wraps=ritz._streamed_projection) as project, \
                patch.object(ritz,'_rotate_in_place',wraps=ritz._rotate_in_place) as rotate:
            result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
        slab,rotated=project.call_args.args[2],rotate.call_args.args[2]
        self.assertEqual((slab.shape,rotated.shape),((257,3),(56,17)))
        self.assertEqual(slab.data.ptr,rotated.data.ptr)
        self.assertEqual(result.algorithm,'streaming_generalized_cholesky_rayleigh_ritz')
        np.testing.assert_allclose(cp.asnumpy(result.eigenvalues),expected,rtol=0,atol=1e-10)
        vectors=cp.asnumpy(result.wavefunctions)
        np.testing.assert_allclose(vectors.T@vectors,np.eye(17),atol=1e-9)

    def test_the_limit_of_one_array_cuts_the_buffer_as_a_budget_of_its_bytes_does(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(89).normal(size=(257,17));x=cp.array(host,order='F')
        limit,budget='PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES','PARSEC_CUPY_STREAMING_RITZ_BYTES'
        # A limit below 1 GiB is the buffer of every basis that it does not hold: 5 columns per slab and 75 rows
        # per tile here, as with the same bytes named as the budget, and the same solve to the last bit.
        runs={}
        for name,other in ((limit,budget),(budget,limit)):
            with patch.dict(os.environ,{name:str(8*257*5)},PARSEC_CUPY_STREAMING_RITZ='1'):
                os.environ.pop(other,None)
                self.assertEqual(ritz._streaming_shape(x),(5,75))
                slab,rotated=ritz._streaming_workspace(x)
                self.assertEqual((slab.shape,rotated.shape,slab.data.ptr),((257,5),(75,17),rotated.data.ptr))
                del slab,rotated
                result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
                self.assertEqual(result.algorithm,'streaming_generalized_cholesky_rayleigh_ritz')
            runs[name]=(cp.asnumpy(result.eigenvalues),cp.asnumpy(result.wavefunctions))
        for mine,theirs in zip(runs[limit],runs[budget]):np.testing.assert_array_equal(mine,theirs)
        # Unset, at its former value and at 2 GiB by name the limit leaves a basis this small one slab and one tile,
        # and its solve the bits it had.
        for value in (None,str(4<<30),str(2<<30)):
            with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='1'):
                for name in (limit,budget):os.environ.pop(name,None)
                if value is not None:os.environ[limit]=value
                self.assertEqual(ritz._streaming_shape(x),(17,257))
                result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
            runs[value]=(cp.asnumpy(result.eigenvalues),cp.asnumpy(result.wavefunctions))
        for value in (str(4<<30),str(2<<30)):
            for mine,theirs in zip(runs[value],runs[None]):np.testing.assert_array_equal(mine,theirs)
        np.testing.assert_allclose(runs[limit][0],runs[None][0],rtol=0,atol=5e-12)

    def test_a_slab_cut_to_a_multiple_leaves_the_tile_the_bytes_of_the_budget_on_the_device(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(101).normal(size=(257,17));x=cp.array(host,order='F')
        # A budget of 6 columns and a multiple of 4: slabs of 4 columns, with a tile of 90 rows by the budget, in a
        # buffer of 1,530 numbers.  A rule counts the 1,542 numbers of the slab of the budget.
        common=dict(PARSEC_CUPY_STREAMING_RITZ='1',PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*257*6),PARSEC_CUPY_RITZ_GRAM_MULTIPLE='4',
                    PARSEC_CUPY_RITZ_DENSE_BACKEND='host',PARSEC_CUPY_RITZ_EIGH_BACKEND='host')
        with patch.dict(os.environ,common),patch.object(ritz,'_rotate_in_place',wraps=ritz._rotate_in_place) as rotate:
            self.assertEqual(ritz._streaming_shape(x),(4,90))
            slab,rotated=ritz._streaming_workspace(x)
            self.assertEqual((slab.shape,rotated.shape,slab.data.ptr),((257,4),(90,17),rotated.data.ptr))
            self.assertEqual((max(slab.size,rotated.size),ritz.streaming_workspace_bytes(257,17)),(1530,8*1542))
            del slab,rotated
            result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
        self.assertEqual(result.algorithm,'streaming_generalized_cholesky_rayleigh_ritz')
        self.assertEqual(tuple(rotate.call_args.args[2].shape),(90,17))
        vectors=cp.asnumpy(result.wavefunctions)
        np.testing.assert_allclose(vectors.T@vectors,np.eye(17),atol=1e-9)

    def test_failed_audit_releases_the_streaming_workspace(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(43).normal(size=(257,17));x=cp.array(host,order='F')
        pool=cp.get_default_memory_pool();used=[]
        def unstable(*args):
            used.append(pool.used_bytes());raise ritz.GeneralizedRitzStabilityError('forced')
        # 3 columns per slab and 56 rows per tile: the buffer of both holds 56*17 = 952 numbers.  The solve
        # allocates from the pool that is read here even if the process allocator has been replaced.
        # Whichever small solve the environment selects fails.
        with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='1',PARSEC_CUPY_STREAMING_RITZ_BYTES='7700'), \
                patch.object(ritz,'solve_whitened_ritz',unstable), \
                patch.object(ritz,'solve_whitened_ritz_on_device',unstable),cp.cuda.using_allocator(pool.malloc):
            # A caller's QR fallback runs inside its handler, where the traceback still refers to the frame of the
            # failed solve; assertRaises would clear that frame before the pool could be read.
            try:ritz.generalized_rayleigh_ritz(self.op,x,consume_basis=True)
            except ritz.GeneralizedRitzStabilityError:used.append(pool.used_bytes())
        # The buffer was in use while the small problem was solved and is free in the handler.
        self.assertEqual(len(used),2);self.assertGreaterEqual(used[0]-used[1],8*952)
        # The basis is rotated only after the audits: the fallback finds it as filtered.
        np.testing.assert_array_equal(cp.asnumpy(x),host)

    def test_streaming_ritz_pieces_match_whole_products(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;rng=np.random.default_rng(23)
        x=cp.array(rng.normal(size=(257,17)),order='F');c=cp.asarray(rng.normal(size=(17,17)))
        whole=cp.asnumpy(x.T@(self.op@x));rotated=cp.asnumpy(x@c)
        # Default slab (the whole small basis), 5 columns per slab, and a single column or row at a time.
        for size in (None,8*257*5,8*17):
            with patch.dict(os.environ):
                os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES',None)
                if size is None:
                    self.assertEqual([ritz._streaming_bytes(n<<30) for n in (0,16,64)],[1<<30,2<<30,4<<30])
                else:os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES']=str(size)
                lower=cp.asnumpy(ritz._streamed_projection(self.op,x))
                np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-12,atol=1e-10)
                moved=x.copy(order='F');pointer=moved.data.ptr;ritz._rotate_in_place(moved,c)
                self.assertEqual(moved.data.ptr,pointer)
                np.testing.assert_allclose(cp.asnumpy(moved),rotated,rtol=0,atol=1e-12)

    def test_column_copy_rotation_writes_the_same_bits(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;rng=np.random.default_rng(37);host=rng.normal(size=(257,17));factors=rng.normal(size=(17,17))
        c=cp.asarray(factors)
        # One tile (the whole small basis), 75 rows per tile with a remainder of 32, and single rows;
        # a row-major basis takes the generic product instead of the direct cuBLAS call.
        for size in (None,8*257*5,8*17):
            for order in ('F','C'):
                moved={}
                for flag in ('0','1'):
                    with patch.dict(os.environ,PARSEC_CUPY_ROTATE_COLUMN_COPY=flag):
                        os.environ.pop('PARSEC_CUPY_STREAMING_RITZ_BYTES',None)
                        if size is not None:os.environ['PARSEC_CUPY_STREAMING_RITZ_BYTES']=str(size)
                        x=cp.array(host,order=order);pointer=x.data.ptr;ritz._rotate_in_place(x,c)
                    self.assertEqual(x.data.ptr,pointer);moved[flag]=cp.asnumpy(x)
                np.testing.assert_array_equal(moved['1'],moved['0'])
                np.testing.assert_allclose(moved['1'],host@factors,rtol=0,atol=1e-12)

    def test_streaming_first_solve_matches_the_generalized_first_solve(self):
        import importlib
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        run_chebff=importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff').run_chebff
        cp=self.cp;settings=ChebFFSettings(polynomial_degree=12,filter_cycles=4)
        common=dict(PARSEC_CUPY_GENERALIZED_RITZ='on',PARSEC_CUPY_MIXED_FILTER='off',PARSEC_CUPY_DISTRIBUTED_FILTER='0',
                    PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1',PARSEC_CUPY_FILTER_GRAPHS='1',PARSEC_CUPY_FILTER_COLUMN_MAJOR='1')
        with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='0',**common):
            expected=run_chebff(self.op,19,settings=settings)
        with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='1',PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*257*5),**common):
            actual=run_chebff(self.op,19,settings=settings)
        self.assertEqual(actual.last_rayleigh_ritz.algorithm,'streaming_generalized_cholesky_rayleigh_ritz')
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues),cp.asnumpy(expected.eigenvalues),rtol=0,atol=5e-11)
        a,b=cp.asnumpy(actual.vectors),cp.asnumpy(expected.vectors)
        np.testing.assert_allclose(np.linalg.svd(a.T@b,compute_uv=False),np.ones(19),atol=5e-9)

    def test_two_array_projection_slabs_match_the_full_product(self):
        import importlib
        from scipy.linalg import eigh
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(29).normal(size=(257,17))
        x=cp.array(host,order='F');applied=cp.asfortranarray(self.op@x)
        whole=host.T@cp.asnumpy(applied)
        # One slab is the former full product; at most 3, 8 and 17 slabs are 6, 3 and 1 columns wide, which
        # makes 3, 6 and 17 slabs of the 17 columns.
        for slabs,width in (('1',17),('3',6),('8',3),('17',1)):
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_GRAM_SLABS=slabs):
                lower=cp.asnumpy(ritz._lower_triangle_product(x,applied))
            np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-12,atol=1e-10)
            for start in range(width,17,width):self.assertFalse(lower[:start,start:start+width].any())
        # The whole two-array solve against the dense generalized eigenproblem and the former full product.
        expected=eigh(np.tril(whole)+np.tril(whole,-1).T,host.T@host,eigvals_only=True)
        with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='0',PARSEC_CUPY_RITZ_GRAM_SLABS='1'):
            former=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'))
        for slabs in ('3','8'):
            with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ='0',PARSEC_CUPY_RITZ_GRAM_SLABS=slabs):
                result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'))
            self.assertEqual(result.algorithm,'generalized_cholesky_rayleigh_ritz')
            np.testing.assert_allclose(cp.asnumpy(result.eigenvalues),expected,rtol=0,atol=1e-10)
            np.testing.assert_allclose(cp.asnumpy(result.eigenvalues),cp.asnumpy(former.eigenvalues),rtol=0,atol=5e-12)
            u,v=cp.asnumpy(former.wavefunctions),cp.asnumpy(result.wavefunctions)
            np.testing.assert_allclose(v.T@v,np.eye(17),atol=1e-9)
            np.testing.assert_allclose(np.linalg.svd(u.T@v,compute_uv=False),np.ones(17),atol=5e-9)

    def test_overlap_slabs_match_dsyrk_and_the_dense_overlap(self):
        import importlib
        from scipy.linalg import eigh
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(31).normal(size=(257,17));x=cp.array(host,order='F')
        whole=host.T@host
        with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='on'):
            syrk=cp.asnumpy(ritz._symmetric_overlap(x))
        np.testing.assert_allclose(np.tril(syrk),np.tril(whole),rtol=1e-12,atol=1e-12)
        # One slab is the full product; at most 3, 8 and 17 slabs are 6, 3 and 1 columns wide, which makes
        # 3, 6 and 17 slabs of the 17 columns.
        for slabs,width in (('1',17),('3',6),('8',3),('17',1)):
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='slabs',PARSEC_CUPY_RITZ_GRAM_SLABS=slabs):
                lower=cp.asnumpy(ritz._symmetric_overlap(x))
            np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.tril(lower),np.tril(syrk),rtol=1e-12,atol=1e-12)
            for start in range(width,17,width):self.assertFalse(lower[:start,start:start+width].any())
        # Whole solves on the two-array and the one-array route against the dense problem and the DSYRK overlap.
        projected=host.T@cp.asnumpy(self.op@x)
        expected=eigh(np.tril(projected)+np.tril(projected,-1).T,whole,eigvals_only=True)
        for streaming in ('0','1'):
            common=dict(PARSEC_CUPY_STREAMING_RITZ=streaming,PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*257*5),
                        PARSEC_CUPY_RITZ_GRAM_SLABS='3')
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='on',**common):
                former=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='slabs',**common):
                result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
            self.assertEqual(result.algorithm.startswith('streaming'),streaming=='1')
            np.testing.assert_allclose(cp.asnumpy(result.eigenvalues),expected,rtol=0,atol=1e-10)
            np.testing.assert_allclose(cp.asnumpy(result.eigenvalues),cp.asnumpy(former.eigenvalues),rtol=0,atol=5e-12)
            u,v=cp.asnumpy(former.wavefunctions),cp.asnumpy(result.wavefunctions)
            np.testing.assert_allclose(v.T@v,np.eye(17),atol=1e-9)
            np.testing.assert_allclose(np.linalg.svd(u.T@v,compute_uv=False),np.ones(17),atol=5e-9)

    def test_a_named_slab_width_cuts_the_overlap_on_the_device(self):
        import importlib
        from scipy.linalg import eigh
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        # 157 columns, as a row block of a shared basis gives them to its overlap: a slab of 128 and one of 29 by
        # the rule of that route, where the count cuts eight slabs of 20.
        cp=self.cp;host=np.random.default_rng(83).normal(size=(257,157));x=cp.array(host,order='F');whole=host.T@host
        with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK='slabs'):
            for name in ('PARSEC_CUPY_RITZ_GRAM_SLABS','PARSEC_CUPY_RITZ_GRAM_MULTIPLE'):os.environ.pop(name,None)
            counted=cp.asnumpy(ritz._symmetric_overlap(x))
            lowers={}
            for width in (128,20,64,157,1000):
                lower=lowers[width]=cp.asnumpy(ritz._symmetric_overlap(x,width))
                np.testing.assert_allclose(np.tril(lower),np.tril(whole),rtol=1e-12,atol=1e-11)
                for start in range(width,157,width):self.assertFalse(lower[:start,start:start+width].any())
                self.assertEqual(bool(lower[0,156]),width>=157)
            # The slabs of the count where the width is theirs, to the last bit; other slabs sum the lower
            # triangle from other pieces.
            np.testing.assert_array_equal(lowers[20],counted)
            np.testing.assert_allclose(np.tril(lowers[128]),np.tril(counted),rtol=0,atol=1e-11)
            # A generalized problem with the overlap of either cut has the same Ritz values at round-off.
            projected=host.T@cp.asnumpy(self.op@x);projected=np.tril(projected)+np.tril(projected,-1).T
            values=[eigh(projected,np.tril(lower)+np.tril(lower,-1).T,eigvals_only=True) for lower in (lowers[128],counted)]
            np.testing.assert_allclose(values[0],values[1],rtol=0,atol=1e-9)
        # DSYRK and the full product have no slabs and read no width.
        for policy in ('on','off'):
            with patch.dict(os.environ,PARSEC_CUPY_RITZ_SYRK=policy):
                np.testing.assert_array_equal(cp.asnumpy(ritz._symmetric_overlap(x,128)),cp.asnumpy(ritz._symmetric_overlap(x)))

    def test_a_sector_on_one_device_forms_its_overlap_in_the_slabs_of_the_count(self):
        import importlib
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        # 157 columns of a basis on one device, as every sector has them on 1, 2 and 4 GPUs with four sectors: eight
        # slabs of 20 by the count, where the rule of a shared row block would cut one of 128 and one of 29.  Neither
        # route of the Ritz step that owns its basis names a width, so its overlap keeps the cut and the bits it had.
        cp=self.cp;host=np.random.default_rng(107).normal(size=(257,157));x=cp.array(host,order='F')
        solve=ritz.solve_whitened_ritz
        for streaming in ('0','1'):
            given=[]
            def kept(overlap,projection):
                given.append(np.array(overlap));return solve(overlap,projection)
            with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ=streaming,PARSEC_CUPY_RITZ_DENSE_BACKEND='host',
                            PARSEC_CUPY_RITZ_EIGH_BACKEND='host'),patch.object(ritz,'solve_whitened_ritz',kept), \
                    patch.object(ritz,'_symmetric_overlap',wraps=ritz._symmetric_overlap) as overlap:
                for name in ('PARSEC_CUPY_RITZ_GRAM_SLABS','PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS',
                             'PARSEC_CUPY_RITZ_GRAM_MULTIPLE','PARSEC_CUPY_RITZ_SYRK','PARSEC_CUPY_STREAMING_RITZ_BYTES'):
                    os.environ.pop(name,None)
                counted=cp.asnumpy(ritz._lower_triangle_product(x,x))
                result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
            self.assertEqual(result.algorithm.startswith('streaming'),streaming=='1')
            # One overlap, asked of the basis alone ...
            self.assertEqual((overlap.call_count,len(overlap.call_args.args),overlap.call_args.kwargs),(1,1,{}))
            # ... and cut as the count cuts it: nothing above the diagonal slabs of 20 columns, the bits of those slabs.
            (lower,)=given
            for start in range(20,157,20):self.assertFalse(lower[:start,start:start+20].any())
            self.assertTrue(lower[19,:20].all())
            np.testing.assert_array_equal(lower,counted)

    def test_gram_multiple_cuts_the_slabs_of_both_routes_on_the_device(self):
        import importlib
        from scipy.linalg import eigh
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(47).normal(size=(257,17));x=cp.array(host,order='F')
        applied=cp.asfortranarray(self.op@x);projected=host.T@cp.asnumpy(applied)
        expected=eigh(np.tril(projected)+np.tril(projected,-1).T,host.T@host,eigvals_only=True)
        name='PARSEC_CUPY_RITZ_GRAM_MULTIPLE'
        # 17 columns in at most 3 slabs of a Gram product and in slabs of the 6 columns of the streaming budget:
        # 6 wide, and 4 with a multiple of 4, which stands for the 64 that no slab of a basis this small reaches.
        common=dict(PARSEC_CUPY_RITZ_GRAM_SLABS='3',PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*257*6),PARSEC_CUPY_RITZ_SYRK='slabs')
        for streaming in ('0','1'):
            runs={}
            for value,width in (('1',6),('4',4),(None,6)):
                with patch.dict(os.environ,PARSEC_CUPY_STREAMING_RITZ=streaming,**common):
                    os.environ.pop(name,None)
                    if value is not None:os.environ[name]=value
                    self.assertEqual((ritz._gram_slab_width(17),ritz._streaming_shape(x)),(width,(width,90)))
                    lower=cp.asnumpy(ritz._lower_triangle_product(x,applied))
                    streamed=cp.asnumpy(ritz._streamed_projection(self.op,x))
                    for product in (lower,streamed):
                        np.testing.assert_allclose(np.tril(product),np.tril(projected),rtol=1e-12,atol=1e-10)
                        for start in range(width,17,width):self.assertFalse(product[:start,start:start+width].any())
                        self.assertTrue(product[width-1,:width].all())
                    result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
                self.assertEqual(result.algorithm.startswith('streaming'),streaming=='1')
                runs[value]=(cp.asnumpy(result.eigenvalues),cp.asnumpy(result.wavefunctions),lower,streamed)
                np.testing.assert_allclose(runs[value][0],expected,rtol=0,atol=1e-10)
            # Unset is the cut of a multiple of 1 here, to the last bit of the Gram matrices and of the solve.
            for mine,theirs in zip(runs[None],runs['1']):np.testing.assert_array_equal(mine,theirs)
            # Other slabs sum the lower triangles from other pieces: the same Ritz pairs to round-off.
            np.testing.assert_allclose(runs['4'][0],runs['1'][0],rtol=0,atol=5e-12)
            np.testing.assert_allclose(np.linalg.svd(runs['1'][1].T@runs['4'][1],compute_uv=False),np.ones(17),atol=5e-9)

    def test_the_serial_control_reads_the_gram_multiple_for_its_projection_only(self):
        import importlib
        from cupy.cuda import cublas
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        ritz=importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        cp=self.cp;host=np.random.default_rng(53).normal(size=(257,17));x=cp.array(host,order='F')
        name='PARSEC_CUPY_RITZ_GRAM_MULTIPLE'
        # The settings of --serial-control, with 17 columns in at most 3 slabs of a Gram product and in slabs of the 6
        # columns of a streaming budget: 6 wide, and 4 with a multiple of 4, which stands for the 64 of a run.  The
        # control takes two arrays and its overlap from DSYRK; a launcher may ask it for one array.
        control=dict(mpi_full_scf._SERIAL_CONTROL_SETTINGS,PARSEC_CUPY_RITZ_GRAM_SLABS='3',
                     PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*257*6))
        self.assertEqual((control['PARSEC_CUPY_RITZ_SYRK'],control['PARSEC_CUPY_STREAMING_RITZ'],control[name]),('on','0','1'))
        solve=ritz.solve_whitened_ritz;below=np.tril_indices(17)
        for streaming,launcher in ((False,{}),(True,dict(PARSEC_CUPY_STREAMING_RITZ='1'))):
            pairs={}
            for multiple,width in (('1',6),('4',4)):
                given=[]
                def kept(overlap,projection):
                    given.append((np.array(overlap),np.array(projection)));return solve(overlap,projection)
                with patch.dict(os.environ,dict(control,**launcher,**{name:multiple})), \
                     patch.object(ritz,'solve_whitened_ritz',kept), \
                     patch.object(cublas,'dsyrk',wraps=cublas.dsyrk) as syrk, \
                     patch.object(ritz,'_gram_slab_width',wraps=ritz._gram_slab_width) as asked:
                    result=ritz.generalized_rayleigh_ritz(self.op,x.copy(order='F'),consume_basis=True)
                self.assertEqual(result.algorithm.startswith('streaming'),streaming)
                # One DSYRK forms the overlap, whose lower triangle no slab cuts.  The width of a slab is asked for
                # the projection of two arrays, once; one array cuts its projection by the streaming budget.
                self.assertEqual((syrk.call_count,asked.call_count),(1,0 if streaming else 1))
                (overlap,projection),=given
                self.assertTrue(overlap[below].all())
                for start in range(width,17,width):self.assertFalse(projection[:start,start:start+width].any())
                self.assertTrue(projection[width-1,:width].all())
                pairs[multiple]=(overlap[below],projection[below])
            # The overlap of the control has the same bits at either multiple; its projection is summed from other
            # slabs.  So the multiple reaches a control through its projection, on either route.
            np.testing.assert_array_equal(pairs['4'][0],pairs['1'][0])
            np.testing.assert_allclose(pairs['4'][1],pairs['1'][1],rtol=1e-12,atol=1e-10)

    def test_device_cg_input_output_and_nondefault_stream(self):
        from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedConjugateGradientBackend
        cp=self.cp;backend=CuPyPreparedConjugateGradientBackend(sp.diags(np.linspace(1,4,257)))
        rhs=np.random.default_rng(12).normal(size=257)
        reference=backend.solve(rhs,np.zeros(257),relative_tolerance=1e-12,absolute_tolerance=1e-14,max_iterations=500)
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:
            device_rhs=cp.asarray(rhs);initial=cp.zeros(257)
            actual=backend.solve(device_rhs,initial,relative_tolerance=1e-12,absolute_tolerance=1e-14,max_iterations=500,device_result=True)
            np.testing.assert_array_equal(cp.asnumpy(actual.pop('solution')),reference.pop('solution'))
        self.assertEqual(actual,reference)

    def test_resident_hartree_matches_full_rhs_chain(self):
        from parsec_python.acceleration.Hartree.cupy_resident import CuPyResidentHartree
        from parsec_python.acceleration.Hartree.cupy_boundary import CuPyMultipoleBoundaryBuilder
        from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedPoissonSolver
        from parsec_python.acceleration.Hartree.symmetry_poisson import SymmetryReducedPoissonSolver
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
        from parsec_python.acceleration.Symmetry import AxisReflectionReduction
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.models import GridSettings,Atom,HartreeSettings
        cp=self.cp
        grid=build_cluster_grid(GridSettings(spacing=.8,radius=3.3,expansion_order=4,shift=(0.,0.,0.)))
        reduction=AxisReflectionReduction.detect(grid,(Atom('H',(0.,0.,0.)),))
        matrix=build_negative_laplacian(grid)
        baseline=SymmetryReducedPoissonSolver(matrix,reduction,solver_factory=CuPyPreparedPoissonSolver)
        device_solver=CuPyPreparedPoissonSolver(baseline.reduced_negative_laplacian)
        builder=CuPyMultipoleBoundaryBuilder(grid,4)
        controls=HartreeSettings(relative_tolerance=1e-12,absolute_tolerance=1e-13)
        resident=CuPyResidentHartree(builder,reduction,device_solver.backend,controls)
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        normalized=np.random.default_rng(71).normal(size=reduction.wedge_size)
        exported=cp.empty((2,reduction.wedge_size),cp.float64)
        resident.export_kernel(((reduction.wedge_size+255)//256,),(256,),
            (np.int32(reduction.wedge_size),resident.ptr,resident.roots,cp.asarray(normalized),cp.asarray(normalized),exported))
        expected_export=SymmetrySCFReducer(reduction).from_full(reduction.expand_vector(normalized)).values
        np.testing.assert_array_equal(cp.asnumpy(exported[0]),expected_export)
        # Projection must follow np.bincount's original per-orbit sum order.
        noisy=np.random.default_rng(9).normal(size=grid.size)
        np.testing.assert_array_equal(cp.asnumpy(resident.project(cp.asarray(noisy))),reduction.reduce_vector(noisy))
        initial=None
        rho=np.exp(-.3*np.sum(grid.coordinates**2,axis=1))
        for factor in (1.,1.03,1.05):
            full=factor*rho
            rhs,_=builder.build(full)
            expected=baseline.solve(rhs,initial,controls,return_wedge=True)
            field=SymmetryScalarField(reduction,full[reduction.representative_rows])
            actual=resident.solve(field,initial)
            self.assertTrue(actual.converged)
            np.testing.assert_allclose(actual.potential.values,expected.potential.values,rtol=0,atol=2e-10)
            np.testing.assert_allclose(actual.right_hand_side.values,expected.right_hand_side.values,rtol=0,atol=2e-13)
            initial=expected.potential

    def test_resident_hartree_chain_on_another_device_returns_the_same_bits(self):
        from functools import partial
        from parsec_python.acceleration.Hartree.cupy_resident import CuPyResidentHartree
        from parsec_python.acceleration.Hartree.cupy_boundary import CuPyMultipoleBoundaryBuilder
        from parsec_python.acceleration.Hartree.cupy_prepared import CuPyPreparedPoissonSolver
        from parsec_python.acceleration.Hartree.symmetry_poisson import SymmetryReducedPoissonSolver
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
        from parsec_python.acceleration.Symmetry import AxisReflectionReduction
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.Laplacian import build_negative_laplacian
        from parsec_python.models import GridSettings,Atom,HartreeSettings
        cp=self.cp
        if cp.cuda.runtime.getDeviceCount()<2:self.skipTest('multiple GPUs required')
        other=cp.cuda.runtime.getDeviceCount()-1
        grid=build_cluster_grid(GridSettings(spacing=.8,radius=3.3,expansion_order=4,shift=(0.,0.,0.)))
        reduction=AxisReflectionReduction.detect(grid,(Atom('H',(0.,0.,0.)),))
        matrix=build_negative_laplacian(grid)
        controls=HartreeSettings(relative_tolerance=1e-12,absolute_tolerance=1e-13)
        def chain(device):
            # CG arrays, boundary geometry and resident maps on one device.
            solver=SymmetryReducedPoissonSolver(matrix,reduction,
                solver_factory=partial(CuPyPreparedPoissonSolver,device_id=device))
            builder=CuPyMultipoleBoundaryBuilder(grid,4,device_id=device)
            return CuPyResidentHartree(builder,reduction,solver.solver.backend,controls)
        first,moved=chain(0),chain(other)
        self.assertEqual((moved.builder.device_id,moved.backend.device_id),(other,other))
        for array in (moved.map,moved.roots,moved.ptr,moved.members):
            self.assertEqual(int(array.device.id),other)
        rho=np.exp(-.3*np.sum(grid.coordinates**2,axis=1))
        initial=None
        for factor in (1.,1.03,1.05):
            field=SymmetryScalarField(reduction,(factor*rho)[reduction.representative_rows])
            expected=first.solve(field,initial)
            actual=moved.solve(field,initial)
            self.assertTrue(expected.converged)
            if initial is None:self.assertGreater(expected.iterations,0)
            np.testing.assert_array_equal(actual.potential.values,expected.potential.values)
            np.testing.assert_array_equal(actual.right_hand_side.values,expected.right_hand_side.values)
            for name in ('converged','iterations','matrix_vector_products','residual_norm','initial_residual_norm'):
                self.assertEqual(getattr(actual,name),getattr(expected,name))
            self.assertEqual(actual.boundary.moments,expected.boundary.moments)
            self.assertEqual(int(cp.cuda.Device().id),0)
            initial=expected.potential
