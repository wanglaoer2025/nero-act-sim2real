"""Additional pre-send rejection only; never modifies an ACT action or old sealed code."""
import numpy as np
from executor.phases import PhaseGuard


class PredictivePhaseGuard(PhaseGuard):
    def inspect(self,current,proposed,widths,previous_width,contact):
        event=super().inspect(current,proposed,widths,previous_width,contact)
        if not contact.close_authorized:
            current=np.asarray(current);proposed=np.asarray(proposed)
            if np.min(proposed[:,2])<max(.045,self.grasp_height-.010):
                raise RuntimeError('Predicted unconfirmed descent below the original grasp floor')
            off_axis=(np.linalg.norm(current[:2]-self.source)>.02
                or np.linalg.norm(proposed[:,:2]-self.source,axis=1).max()>.02)
            if off_axis and np.min(proposed[:,2])<min(current[2],self.grasp_height+.065)-1e-9:
                raise RuntimeError('Predicted low descent before alignment; no action sent')
        return event
