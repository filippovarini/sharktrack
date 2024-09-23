import unittest
from time_processor import ms_to_string, string_to_ms

class TestGenerateCorrectOutput(unittest.TestCase):
    def test__convert_time_correctly(self):
        matches = [
            (1000, "00h:00m:01s:000ms"),
            (1001, "00h:00m:01s:001ms"),
            (3600000, "01h:00m:00s:000ms"),
            (3600200, "01h:00m:00s:200ms"),
            (2335, "00h:00m:02s:335ms")
        ]

        for ms, ms_string in matches:
            self.assertEqual(ms_to_string(ms), ms_string)    
            self.assertEqual(ms_to_string(ms), ms_string)    
            print(ms)

    
    


if __name__ == '__main__':
    unittest.main()