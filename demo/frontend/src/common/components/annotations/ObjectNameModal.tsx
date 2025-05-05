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
import {Button, Input} from 'react-daisyui';
import {useEffect, useRef} from 'react';

type Props = {
  isOpen: boolean;
  onConfirm: (name: string) => void;
  onCancel: () => void;
  value: string;
  onChange: (value: string) => void;
  // Add a prop to customize the title based on context
  modalTitle?: string;
  confirmButtonText?: string;
};

export default function ObjectNameModal({
  isOpen,
  onConfirm,
  onCancel,
  value,
  onChange,
  modalTitle = 'Name your object', // Default title
  confirmButtonText = 'Confirm', // Default button text
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      // Slight delay to ensure modal is fully rendered before focusing
      const timer = setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select(); // Select existing text when editing
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    onConfirm(value.trim());
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    // Stop propagation to prevent keys typed here from triggering global listeners
    event.stopPropagation();

    if (event.key === 'Enter') {
      event.preventDefault(); // Prevent form submission if inside a form
      onConfirm(value.trim());
    } else if (event.key === 'Escape') {
      // No need to prevent default for Escape, but stopPropagation is good
      onCancel();
    }
    // No specific action needed for 'k' here, but stopPropagation() above handles it.
  }

  if (!isOpen) {
    return null;
  }

  return (
    // Use a more specific class for potential styling overrides
    <div className="object-name-modal modal modal-open">
      {/* Use modal-box for standard DaisyUI modal styling */}
      <div className="modal-box bg-black text-white">
        <h3 className="font-bold text-lg">{modalTitle}</h3>
        <p className="py-4 text-gray-400">
          {modalTitle === 'Name your object'
            ? 'Enter a name for the new object, or leave it blank for a default name.'
            : 'Enter a new name for the object.'}
        </p>
        {/* Use form for better accessibility and submission handling */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <Input
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown} // Ensure this handler is used
            placeholder="e.g., Car, Person"
            // Consistent styling classes
            className="w-full bg-graydark-800 text-white border-graydark-700 focus:outline-none focus:border-blue-500"
            autoFocus
          />
          <div className="modal-action mt-2">
            <Button
              type="button" // Ensure it doesn't submit the form by default
              color="ghost"
              onClick={onCancel}
              className="text-white hover:bg-graydark-700">
              Cancel
            </Button>
            <Button
              type="submit" // This button triggers form submission
              color="primary"
              className="bg-blue-500 text-white hover:bg-blue-600">
              {confirmButtonText}
            </Button>
          </div>
        </form>
      </div>
      {/* Optional: Add an overlay that closes the modal on click */}
      <div className="modal-backdrop bg-black bg-opacity-30" onClick={onCancel}></div>
    </div>
  );
}
