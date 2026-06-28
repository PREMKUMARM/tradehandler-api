"""Entry bar close-position filter (avoid buying spike tops)."""
import os
import unittest

from services.entry_quality import bar_close_position, entry_bar_quality_ok


class TestEntryClosePosition(unittest.TestCase):
    def test_bar_close_position(self):
        self.assertAlmostEqual(bar_close_position(10.0, 20.0, 15.0), 0.5)
        self.assertAlmostEqual(bar_close_position(13.5, 20.5, 20.4), 0.986, places=2)

    def test_rejects_close_near_high(self):
        os.environ["ENTRY_MAX_CLOSE_POSITION_IN_BAR"] = "0.75"
        ok, why = entry_bar_quality_ok(
            kind="PE",
            bar_open=15.9,
            bar_close=20.4,
            spot=22500.0,
            prev_close=22400.0,
            spot_prev=22480.0,
            bar_high=20.5,
            bar_low=13.5,
        )
        self.assertFalse(ok)
        self.assertEqual(why, "close_near_high")

    def test_allows_mid_range_close(self):
        os.environ["ENTRY_MAX_CLOSE_POSITION_IN_BAR"] = "0.75"
        ok, why = entry_bar_quality_ok(
            kind="CE",
            bar_open=17.0,
            bar_close=19.0,
            spot=24000.0,
            prev_close=23900.0,
            spot_prev=23950.0,
            bar_high=22.0,
            bar_low=16.0,
        )
        self.assertTrue(ok)
        self.assertEqual(why, "")

    def test_disabled_when_zero(self):
        os.environ["ENTRY_MAX_CLOSE_POSITION_IN_BAR"] = "0"
        ok, why = entry_bar_quality_ok(
            kind="PE",
            bar_open=15.9,
            bar_close=20.4,
            spot=22500.0,
            prev_close=22400.0,
            bar_high=20.5,
            bar_low=13.5,
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
