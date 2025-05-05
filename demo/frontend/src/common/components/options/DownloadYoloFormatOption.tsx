/**
 * Option Button to download the full YOLO dataset (images, labels, classes) as a Zip archive.
 * This corresponds to the "Download All" button.
 */
import { Archive } from '@carbon/icons-react'; // Example icon
import { useState } from 'react';
import { useAtomValue } from 'jotai';
import { sessionAtom } from '@/demo/atoms';
import { downloadYoloFormatZip } from '@/common/api/downloadApi';
import OptionButton from './OptionButton';
import Logger from '@/common/logger/Logger';

export default function DownloadYoloFormatOption() {
  const session = useAtomValue(sessionAtom);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    // Prevent action if already loading
    if (isLoading) {
      Logger.warn('Download All cancelled: Already loading.');
      return;
    }
    // Check for session ID *before* starting the download
    if (!session?.session_id) {
      Logger.warn('Download All cancelled: No session ID available yet.');
      // Optional: Add user feedback like an alert or snackbar here
      alert('Session data is not yet available. Please wait for processing to complete.');
      return;
    }

    setIsLoading(true);
    try {
      await downloadYoloFormatZip(session.session_id);
    } catch (error) {
      // Error already logged and alerted in API function
      Logger.error('Download All Option caught error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <OptionButton
      title="Download All (YOLO Zip)" // Button text
      Icon={Archive} // Choose an appropriate icon
      loadingProps={{
        loading: isLoading,
        label: 'Downloading...',
      }}
      onClick={handleClick}
    />
  );
}