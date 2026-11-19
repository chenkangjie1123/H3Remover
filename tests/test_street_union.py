import unittest
import numpy as np
from pipeline.street_union import proposals, choose_masks
from pipeline.common import collect_instances

class StreetUnionTests(unittest.TestCase):
    def test_static_one_frame_track_kept_without_motion(self):
        mask=np.ones((1,4,6),bool)
        class Predictor:
            def handle_stream_request(self, request):
                return [dict(frame_index=0,outputs=dict(out_obj_ids=[1],out_binary_masks=mask))]
        result,_=collect_instances(Predictor(),0,np.zeros((1,4,6)),4,6,threshold=.5,fraction=.5,
                                   min_frames=2,min_pixels=32,max_objects=16,direct=True)
        np.testing.assert_array_equal(result,mask)

    def test_motion_outside_direct_generates_category_free_prompt(self):
        p=np.zeros((30,40),np.float32);p[1:9,1:9]=1;p[12:22,20:30]=1
        d=np.zeros_like(p,bool);d[1:9,1:9]=True
        items=list(proposals(p,d));self.assertEqual(len(items),1)
        seed,box=items[0];self.assertAlmostEqual(box[0],25/40)
        chosen=choose_masks([np.ones_like(d),seed,d],seed,d)
        np.testing.assert_array_equal(chosen,seed)
        self.assertFalse((chosen&d).any())
