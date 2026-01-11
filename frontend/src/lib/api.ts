/**
 * API Service for V2 Endpoints
 * Handles communication with the advanced AI try-on backend
 */

const API_BASE = '/api';
const API_V2_BASE = '/api/v2';

// Types
export interface HardwareProfile {
  system: string;
  arch: string;
  device_type: string;
  device_name: string;
  vram_gb: number;
  compute_backend: string;
  model_profile: string;
  recommended_batch_size: number;
  recommended_steps: number;
  estimated_speed: string;
}

export interface SystemStatus {
  status: string;
  hardware: HardwareProfile | null;
  ai_engine: boolean;
  comfyui: boolean;
  models_loaded: string[];
  version: string;
}

export interface ModelProfile {
  id: string;
  name: string;
  face_embedding: number[] | null;
  skin_colors: Record<string, [number, number, number]>;
  body_shape: string;
  reference_images: string[];
  created_at: string;
}

export interface TryOnOptions {
  preserve_face: boolean;
  preserve_skin_tone: boolean;
  upscale: boolean;
  face_restore: boolean;
  steps: number;
  denoise: number;
  seed: number;
  prompt: string;
  negative_prompt: string;
}

export interface TryOnResult {
  success: boolean;
  result_url?: string;
  result_id?: string;
  seed?: number;
  generation_time?: number;
  error?: string;
}

export interface FeatureExtractionResult {
  success: boolean;
  face_detected: boolean;
  body_detected: boolean;
  skin_colors: Record<string, [number, number, number]>;
  body_shape: string;
  pose_type: string;
  error?: string;
}

// API Functions

/**
 * Check if V2 API is available
 */
export async function checkV2Available(): Promise<boolean> {
  try {
    const response = await fetch(`${API_V2_BASE}/system/status`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get system status including hardware info
 */
export async function getSystemStatus(): Promise<SystemStatus | null> {
  try {
    const response = await fetch(`${API_V2_BASE}/system/status`);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Detect hardware capabilities
 */
export async function detectHardware(): Promise<HardwareProfile | null> {
  try {
    const response = await fetch(`${API_V2_BASE}/system/detect-hardware`, {
      method: 'POST',
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Get all model profiles
 */
export async function getProfiles(): Promise<ModelProfile[]> {
  try {
    const response = await fetch(`${API_V2_BASE}/profiles`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.profiles || [];
  } catch {
    return [];
  }
}

/**
 * Create a new model profile from an image
 */
export async function createProfile(
  image: File,
  name: string
): Promise<ModelProfile | null> {
  try {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('name', name);

    const response = await fetch(`${API_V2_BASE}/profiles`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Delete a model profile
 */
export async function deleteProfile(profileId: string): Promise<boolean> {
  try {
    const response = await fetch(`${API_V2_BASE}/profiles/${profileId}`, {
      method: 'DELETE',
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Extract features from an image
 */
export async function extractFeatures(
  image: File
): Promise<FeatureExtractionResult | null> {
  try {
    const formData = new FormData();
    formData.append('image', image);

    const response = await fetch(`${API_V2_BASE}/extract-features`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Process a garment image (remove background, classify)
 */
export async function processGarment(
  image: File,
  removeBackground: boolean = true
): Promise<{
  processed_url: string;
  mask_url: string;
  garment_info: Record<string, unknown>;
} | null> {
  try {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('remove_background', removeBackground.toString());

    const response = await fetch(`${API_V2_BASE}/process-garment`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Generate advanced try-on with V2 API
 */
export async function generateAdvancedTryon(
  personImage: File,
  options: Partial<TryOnOptions> & {
    garmentImage?: File;
    maskImage?: File;
    profileId?: string;
  }
): Promise<TryOnResult> {
  try {
    const formData = new FormData();
    formData.append('person_image', personImage);

    if (options.garmentImage) {
      formData.append('garment_image', options.garmentImage);
    }
    if (options.maskImage) {
      formData.append('mask_image', options.maskImage);
    }
    if (options.profileId) {
      formData.append('profile_id', options.profileId);
    }

    // Add optional parameters
    if (options.preserve_face !== undefined) {
      formData.append('preserve_face', options.preserve_face.toString());
    }
    if (options.preserve_skin_tone !== undefined) {
      formData.append('preserve_skin_tone', options.preserve_skin_tone.toString());
    }
    if (options.upscale !== undefined) {
      formData.append('upscale', options.upscale.toString());
    }
    if (options.face_restore !== undefined) {
      formData.append('face_restore', options.face_restore.toString());
    }
    if (options.steps !== undefined) {
      formData.append('steps', options.steps.toString());
    }
    if (options.denoise !== undefined) {
      formData.append('denoise', options.denoise.toString());
    }
    if (options.seed !== undefined) {
      formData.append('seed', options.seed.toString());
    }
    if (options.prompt) {
      formData.append('prompt', options.prompt);
    }
    if (options.negative_prompt) {
      formData.append('negative_prompt', options.negative_prompt);
    }

    const response = await fetch(`${API_V2_BASE}/tryon/advanced`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data.detail || 'Generation failed',
      };
    }

    return {
      success: true,
      result_url: data.result_url,
      result_id: data.result_id,
      seed: data.seed,
      generation_time: data.generation_time,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Batch try-on with multiple garments
 */
export async function batchTryon(
  personImage: File,
  garmentImages: File[],
  options?: Partial<TryOnOptions>
): Promise<TryOnResult[]> {
  try {
    const formData = new FormData();
    formData.append('person_image', personImage);

    garmentImages.forEach((garment) => {
      formData.append('garment_images', garment);
    });

    // Add options
    if (options) {
      Object.entries(options).forEach(([key, value]) => {
        if (value !== undefined) {
          formData.append(key, value.toString());
        }
      });
    }

    const response = await fetch(`${API_V2_BASE}/batch-tryon`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      return [{ success: false, error: data.detail || 'Batch generation failed' }];
    }

    return data.results || [];
  } catch (error) {
    return [
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
    ];
  }
}

// Legacy API functions (for fallback)

/**
 * Generate try-on using legacy ComfyUI API
 */
export async function generateLegacyTryon(
  personImage: File,
  maskImage: File,
  options: {
    garmentImage?: File;
    prompt: string;
    negativePrompt: string;
    steps: number;
    cfgScale: number;
    sampler: string;
    denoise: number;
    model: string;
  }
): Promise<{ job_id: string } | null> {
  try {
    const formData = new FormData();
    formData.append('person_image', personImage);
    formData.append('mask_image', maskImage);

    if (options.garmentImage) {
      formData.append('garment_image', options.garmentImage);
    }

    formData.append('prompt', options.prompt);
    formData.append('negative_prompt', options.negativePrompt);
    formData.append('steps', options.steps.toString());
    formData.append('cfg_scale', options.cfgScale.toString());
    formData.append('sampler', options.sampler);
    formData.append('denoise', options.denoise.toString());
    formData.append('model', options.model);

    const response = await fetch(`${API_BASE}/tryon`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Poll for legacy try-on result
 */
export async function pollLegacyResult(
  jobId: string
): Promise<{
  status: string;
  progress?: number;
  result_url?: string;
  result_id?: string;
  error?: string;
} | null> {
  try {
    const response = await fetch(`${API_BASE}/tryon/${jobId}`);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

/**
 * Check system health
 */
export async function checkHealth(): Promise<{
  status: string;
  models?: Array<{ name: string; file: string }>;
} | null> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}
