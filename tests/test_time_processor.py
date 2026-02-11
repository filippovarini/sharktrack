import unittest
from utils.time_processor import ms_to_string, string_to_ms

class TestGenerateCorrectOutput(unittest.TestCase):
    def test_ms_to_string(self):
        # Basic tests
        self.assertEqual(ms_to_string(0), "00h:00m:00s:000ms")
        self.assertEqual(ms_to_string(1000), "00h:00m:01s:000ms")
        self.assertEqual(ms_to_string(61000), "00h:01m:01s:000ms")
        self.assertEqual(ms_to_string(3600000), "01h:00m:00s:000ms")
        self.assertEqual(ms_to_string(3661000), "01h:01m:01s:000ms")

        # Edge cases
        self.assertEqual(ms_to_string(1), "00h:00m:00s:001ms")
        self.assertEqual(ms_to_string(999), "00h:00m:00s:999ms")
        self.assertEqual(ms_to_string(86399999), "23h:59m:59s:999ms")  # Max time in a day

    def test_string_to_ms(self):
        # Basic tests
        self.assertEqual(string_to_ms("00h:00m:00s:000ms"), 0)
        self.assertEqual(string_to_ms("00h:00m:01s:000ms"), 1000)
        self.assertEqual(string_to_ms("00h:01m:01s:000ms"), 61000)
        self.assertEqual(string_to_ms("01h:00m:00s:000ms"), 3600000)
        self.assertEqual(string_to_ms("01h:01m:01s:000ms"), 3661000)

        # Edge cases
        self.assertEqual(string_to_ms("00h:00m:00s:001ms"), 1)
        self.assertEqual(string_to_ms("00h:00m:00s:999ms"), 999)
        self.assertEqual(string_to_ms("23h:59m:59s:999ms"), 86399999)

    
if __name__ == '__main__':
    unittest.main()