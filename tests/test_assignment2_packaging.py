"""Packaging must preserve frozen evidence, including epoch-dated receipts."""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from assignment2.package_submission import create_archives


class PackagingTests(unittest.TestCase):
    def test_epoch_receipt_is_packaged_without_changing_its_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'frozen'
            receipt = source / 'assignment2/deployed_version.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text('{"frozen": true}\n')
            os.utime(receipt, (0, 0))
            campaign = root / 'campaign'
            for model in ['openai_mini', 'qwen_7b_local']:
                run = campaign / model / 'final'
                run.mkdir(parents=True)
                (run / 'manifest.json').write_text(json.dumps({'source_files': {'code.py': 'hash'}}))
            (campaign / 'ANALYSIS_COMPLETE').write_text('analysis passed\n')
            materials = root / 'materials'
            materials.mkdir()
            (materials / 'PARTNER_GUIDE.md').write_text('Human report still required.\n')
            with patch('assignment2.package_submission.source_provenance'), \
                 patch('assignment2.package_submission.validate_run_directory'), \
                 patch('assignment2.package_submission.execution_source_hashes', return_value={'code.py': 'hash'}), \
                 patch('assignment2.package_submission.release_files', return_value=[]):
                archives = create_archives(campaign, source, root / 'output', materials)
                self.assertEqual(len(archives), 3)
                for archive_path in archives:
                    with zipfile.ZipFile(archive_path) as archive:
                        self.assertIsNone(archive.testzip())
                        self.assertEqual(archive.read('assignment2/deployed_version.json'), receipt.read_bytes())
                        self.assertEqual(archive.getinfo('assignment2/deployed_version.json').date_time[0], 1980)
                        self.assertIn('assignment2/report_materials/PARTNER_GUIDE.md', archive.namelist())
                self.assertFalse(list((root / 'output').glob('*.tmp')))
                self.assertEqual(receipt.stat().st_mtime, 0)
                bad = campaign / 'openai_mini/final/manifest.json'
                bad.write_text(json.dumps({'source_files': {'code.py': 'different'}}))
                with self.assertRaisesRegex(ValueError, 'source differs'):
                    create_archives(campaign, source, root / 'rejected')
                self.assertFalse((root / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
