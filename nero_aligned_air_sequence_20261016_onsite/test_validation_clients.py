from types import SimpleNamespace
import unittest

from full_runtime import FullSession


class ValidationClientTests(unittest.TestCase):
    def session(self, available=True):
        s = FullSession.__new__(FullSession)
        s.report = {}; created = []; destroyed = []
        def create(kind, service):
            c = SimpleNamespace(wait_for_service=lambda timeout_sec: available,
                                call_async=lambda request: request)
            created.append((kind, service, c))
            return c
        s.node = SimpleNamespace(create_client=create, destroy_client=destroyed.append)
        s.wait = lambda response: response
        return s, created, destroyed

    def test_ten_state_checks_reuse_one_client_without_reusing_results(self):
        s, created, destroyed = self.session()
        replies = [object() for _ in range(10)]
        for reply in replies:
            self.assertIs(s.call('/check_state_validity', object, reply), reply)
        self.assertEqual(len(created), 1)
        self.assertEqual(destroyed, [])
        self.assertEqual(len(s.report['validation_service_calls']), 10)

    def test_scene_and_state_checks_have_separate_clients(self):
        s, created, _ = self.session()
        s.call('/get_planning_scene', object, object())
        s.call('/check_state_validity', object, object())
        s.call('/get_planning_scene', object, object())
        self.assertEqual(len(created), 2)

    def test_unavailable_service_is_destroyed_and_still_rejected(self):
        s, created, destroyed = self.session(False)
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            s.call('/check_state_validity', object, object())
        self.assertEqual(destroyed, [created[0][2]])

    def test_invalid_state_response_is_never_replaced_by_previous_valid_result(self):
        s, _, _ = self.session()
        self.assertTrue(s.call('/check_state_validity', object, SimpleNamespace(valid=True)).valid)
        self.assertFalse(s.call('/check_state_validity', object, SimpleNamespace(valid=False)).valid)


if __name__ == '__main__':
    unittest.main()
