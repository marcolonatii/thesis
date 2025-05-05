/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import {BaseTracklet} from '@/common/tracker/Tracker';
import {trackletNamesAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

// Export the type
export type ExtendedTracklet = BaseTracklet & { name?: string | null };

/**
 * Generates a display label for a tracklet.
 * It uses the name stored in trackletNamesAtom if available,
 * otherwise defaults to "Object {id + 1}".
 *
 * @param tracklet The tracklet object.
 * @param trackletNames A record mapping tracklet IDs to their names.
 * @returns The display label string.
 */
function getObjectLabelInternal(tracklet: ExtendedTracklet, trackletNames: Record<number, string>): string {
  // Ensure tracklet and tracklet.id are valid before accessing
  if (!tracklet || typeof tracklet.id !== 'number') {
    return 'Invalid Object'; // Or handle appropriately
  }
  return trackletNames[tracklet.id] || `Object ${tracklet.id + 1}`;
}


// Custom hook to get the label, encapsulating Jotai logic
export function useObjectLabel(tracklet: ExtendedTracklet): string {
    const trackletNames = useAtomValue(trackletNamesAtom);
    return getObjectLabelInternal(tracklet, trackletNames);
}

// Non-hook version for use outside components if needed, requires passing names
export function getObjectLabel(tracklet: ExtendedTracklet, trackletNames: Record<number, string>): string {
  return getObjectLabelInternal(tracklet, trackletNames);
}