/**
 * Option Button to download YOLO format labels (labels/*.txt + classes.txt) as a Zip archive.
 * This corresponds to the "Download Boxes" button.
 */
import { Data_1 } from '@carbon/icons-react'; // Example icon
import { useState } from 'react';
import { useAtomValue } from 'jotai';
import { sessionAtom } from '@/demo/atoms';
import { downloadYoloLabelsZip } from '@/common/api/downloadApi';
import OptionButton from './OptionButton';
import Logger from '@/common/logger/Logger';

export default function DownloadYoloLabelsOption() {
  const session = useAtomValue(sessionAtom);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    // Prevent action if already loading
    if (isLoading) {
      Logger.warn('Download Boxes cancelled: Already loading.');
      return;
    }
    // Check for session ID *before* starting the download
    if (!session?.session_id) {
      Logger.warn('Download Boxes cancelled: No session ID available yet.');
      // Optional: Add user feedback like an alert or snackbar here
      alert('Session data is not yet available. Please wait for processing to complete.');
      return;
    }

    setIsLoading(true);
    try {
      await downloadYoloLabelsZip(session.session_id);
    } catch (error) {
      // Error already logged and alerted in API function
      Logger.error('Download Boxes Option caught error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <OptionButton
      title="Download Boxes (YOLO Zip)" // Button text
      Icon={Data_1} // Choose an appropriate icon
      loadingProps={{
        loading: isLoading,
        label: 'Downloading...',
      }}
      onClick={handleClick}
    />
  );
}