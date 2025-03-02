
import { UserSettings } from '@/utils/types';

// Local storage key for user settings
const USER_SETTINGS_KEY = 'userSettings';

// Default user settings
const DEFAULT_USER_SETTINGS: UserSettings = {
  phoneNumber: '',
  whatsappEnabled: true
};

// Get user settings from localStorage
export function getUserSettings(): UserSettings {
  try {
    const storedSettings = localStorage.getItem(USER_SETTINGS_KEY);
    if (storedSettings) {
      return JSON.parse(storedSettings) as UserSettings;
    }
    return DEFAULT_USER_SETTINGS;
  } catch (error) {
    console.error('Error getting user settings:', error);
    return DEFAULT_USER_SETTINGS;
  }
}

// Save user settings to localStorage
export function saveUserSettings(settings: UserSettings): boolean {
  try {
    localStorage.setItem(USER_SETTINGS_KEY, JSON.stringify(settings));
    return true;
  } catch (error) {
    console.error('Error saving user settings:', error);
    return false;
  }
}

// Update user phone number
export function updateUserPhoneNumber(phoneNumber: string): boolean {
  try {
    const currentSettings = getUserSettings();
    const updatedSettings = {
      ...currentSettings,
      phoneNumber
    };
    return saveUserSettings(updatedSettings);
  } catch (error) {
    console.error('Error updating user phone number:', error);
    return false;
  }
}

// Toggle WhatsApp notifications
export function toggleWhatsAppNotifications(enabled: boolean): boolean {
  try {
    const currentSettings = getUserSettings();
    const updatedSettings = {
      ...currentSettings,
      whatsappEnabled: enabled
    };
    return saveUserSettings(updatedSettings);
  } catch (error) {
    console.error('Error toggling WhatsApp notifications:', error);
    return false;
  }
}

// Check if user has a valid phone number for WhatsApp
export function hasValidPhoneNumber(): boolean {
  const settings = getUserSettings();
  return !!settings.phoneNumber && settings.phoneNumber.length > 8;
}

// Format phone number to international format (required for WhatsApp API)
export function formatPhoneNumber(phoneNumber: string): string {
  // Remove all non-digit characters
  let cleaned = phoneNumber.replace(/\D/g, '');
  
  // Add + prefix if not present
  if (!cleaned.startsWith('+')) {
    // Assume Turkish number if no country code
    if (!cleaned.startsWith('90') && cleaned.length <= 10) {
      cleaned = '90' + cleaned;
    }
    cleaned = '+' + cleaned;
  }
  
  return cleaned;
}
