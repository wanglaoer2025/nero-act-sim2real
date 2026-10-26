"""Live short-prefix dispatch and lazy ROS transport; no motion on import.

The camera/semantic frontend and a receive-only restorer must be qualified before
these components can be assembled for hardware. Nominal simulation never grants
that qualification.
"""
import time

import numpy as np

from calibration.bundle import SCHEMA, calibration_hash
from executor.visual_observation import require_visual_observation


def require_deployment_evidence(calibration, evaluation, receipt, *, now_wall_s):
    if (calibration.get('schema') != SCHEMA or calibration.get('verified') is not True
            or calibration.get('synthetic') is not False
            or calibration.get('calibration_sha256') != calibration_hash(calibration)):
        raise ValueError('Independent real calibration required; nominal training is insufficient')
    if (evaluation.get('passed') != 16 or evaluation.get('total') != 16
            or not all(evaluation.get(k) is True for k in ['complete','finalized',
                'measured_geometry','live_frontend_qualified','end_to_end_timing_verified',
                'all_parameters_predeclared_before_training'])
            or evaluation.get('calibration_sha256') != calibration['calibration_sha256']
            or not evaluation.get('checkpoint_sha256') or not evaluation.get('source_sha256')):
        raise ValueError('All16 qualified complete-task cases, calibration and source/model binding required')
    if (receipt.get('mode') != 'visual15_complete_ACT_once'
            or receipt.get('checkpoint_sha256') != evaluation['checkpoint_sha256']
            or receipt.get('calibration_sha256') != calibration['calibration_sha256']
            or receipt.get('source_sha256') != evaluation['source_sha256']
            or not all(receipt.get(k) is True for k in ['operator_at_estop','empty_cup',
                    'swept_area_clear','single_controller'])
            or not 0 <= now_wall_s-float(receipt.get('confirmed_wall_s',float('-inf'))) <= 300):
        raise ValueError('Current scene and one-shot visual15 authorization required; old replay receipts rejected')


class VisualDispatcher:
    def __init__(self, core, transport, *, clock=time.monotonic):
        self.core,self.transport,self.clock=core,transport,clock
        self._stop_sent=False

    def stop(self):
        self.core.stop()
        if not self._stop_sent:
            self.transport.latch_stop();self._stop_sent=True

    def await_prediction(self, future, *, timeout_s=5.):
        deadline=self.clock()+timeout_s
        try:
            while not future.done():
                self.transport.health()
                if self.clock() >= deadline:
                    raise TimeoutError('Live visual prediction timeout')
                self.transport.spin(.002)
            self.transport.health()
            return future.result()
        except BaseException:
            future.cancel();self.stop();raise

    def dispatch_prediction(self, packet, *, phase='approach'):
        from executor.visual_inference import verify_prediction_packet
        try:
            verify_prediction_packet(packet)
            if packet['details']['previous_gripper_target_m'] != float(self.core.history[7]):
                raise ValueError('Prediction history no longer matches the last actually submitted target')
            return self.dispatch(packet['raw'],packet['frame'],packet['geometry'],
                packet['paired_feedback'],phase=phase)
        except BaseException:
            self.stop();raise

    def dispatch(self, raw, frame, geometry, paired_feedback, *, phase='approach'):
        try:
            self.transport.health()
            packet=self.core.prepare(raw,frame,geometry,paired_feedback,now_s=self.clock(),phase=phase)
            self.transport.validate(packet['rows'])
            # Collision services can consume the camera/feedback freshness budget.
            require_visual_observation(frame,geometry,paired_feedback,now_s=self.clock(),calibration=self.core.calibration)
            self.transport.health()
            result=self.transport.submit(packet['rows'],packet['request_id'])
            self.core.commit(packet['request_id'],packet['rows'],sent_s=result['sent_s'],
                ack_s=result['ack_s'],accepted=result['accepted'])
            self.transport.complete(result['results'],min(frame.exposure_monotonic_s+.75,
                result['sent_s']+len(packet['rows'])*.05+.25))
            return packet
        except BaseException:
            self.stop();raise

    def close(self):
        errors=[];readonly=False
        for name,action in [('stop',self.stop),('cancel',self.transport.cancel_all),
                            ('readonly',self.transport.restore_readonly)]:
            try:
                action()
                if name=='readonly':readonly=True
            except BaseException as exc:errors.append(dict(stage=name,error=str(exc)))
        return dict(stopped=self.core.stopped,readonly_restored=readonly,
            automatic_retry=False,automatic_return=False,cleanup_errors=errors)


class RosVisualTransport:
    """Wrap existing guarded ActSession after all qualification checks pass."""
    def __init__(self, session, clients, *, clock=time.monotonic, restore_readonly=None):
        self.session,self.clients,self.clock=session,clients,clock
        self.restorer=restore_readonly

    def health(self): self.session.guard()
    def spin(self, seconds): self.session.spin(seconds)
    def validate(self, rows): self.session.validate_rows(rows)

    def submit(self, rows, request_id):
        rows=np.asarray(rows,float)
        if rows.shape not in [(1,8),(2,8)] or not np.isfinite(rows).all():
            raise ValueError('ROS visual transport only accepts current1/2-point ACT prefixes')
        sent=self.clock()
        futures=self.session.send_block(self.clients,rows)
        return dict(request_id=request_id,sent_s=sent,ack_s=self.clock(),accepted=True,results=futures)

    def complete(self, results, deadline):
        def check():
            self.health()
            if self.clock() >= deadline:
                raise TimeoutError('Both controller results must arrive within the observation tail deadline')
        check()
        while not all(f.done() for f in results):
            check()
            for f in results:
                if f.done():self.require_result(f)
            self.spin(.002)
        for f in results:self.require_result(f)
        check()

    @staticmethod
    def require_result(future):
        if future.exception() is not None: raise RuntimeError('Controller result future failed') from future.exception()
        response=future.result()
        if response.status != 4 or response.result.error_code != 0:
            raise RuntimeError('Arm/gripper controller result failed')

    def latch_stop(self):
        self.session.close_control_gate()
        if self.session.report.get('act_stop_latched') is not True:
            raise RuntimeError('Real control gate did not acknowledge latched stop')

    def cancel_all(self): self.session.cancel_all()

    def restore_readonly(self):
        if self.restorer is None:
            raise RuntimeError('Receive-only driver restorer is not qualified/configured')
        result=self.restorer()
        if result.get('runtime_send_blocked') is not True or result.get('can_tx_unchanged') is not True:
            raise RuntimeError('Receive-only restoration or CAN TX check failed')
