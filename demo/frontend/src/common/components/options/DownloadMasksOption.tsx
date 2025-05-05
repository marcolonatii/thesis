/**
 * Option Button to download segmentation masks as JSON.
 */
import { DocumentDownload } from '@carbon/icons-react';
import { useState } from 'react';
import { useAtomValue } from 'jotai';
import { sessionAtom } from '@/demo/atoms';
import { downloadMasksJson } from '@/common/api/downloadApi';
import OptionButton from './OptionButton';
import Logger from '@/common/logger/Logger';


export default function DownloadMasksOption() {
  const session = useAtomValue(sessionAtom);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    if (!session?.session_id || isLoading) {
        Logger.warn('Download Masks cancelled: No session ID or already loading.');
        alert('Session data is not yet available. Please wait for processing to complete.');
        return;
    }
    setIsLoading(true);
    try {
      await downloadMasksJson(session.session_id);
    } catch (error) {
      Logger.error('Download Masks Option caught error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <OptionButton
      title="Download Masks (JSON)"
      Icon={DocumentDownload} // Choose an appropriate icon
      loadingProps={{
        loading: isLoading,
        label: 'Downloading...',
      }}
      onClick={handleClick}
    />
  );
}