"""Describe only this patch's action integrity; preserve and disclose existing holds."""


def action_provenance():
    return dict(contact_escape_rewrites_targets=False,
        existing_contact_arm_holds_retained=True,
        existing_pregrasp_gripper_holds_retained=True,
        teacher_generated_task_action_fallback=False)
