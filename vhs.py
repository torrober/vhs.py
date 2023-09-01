import math
import random
import numpy as np
import cv2
import blendmodes
import copy


class VHS:
    def __init__(self, lumaCompressionRate, lumaNoiseSigma, lumaNoiseMean, chromaCompressionRate, chromaNoiseIntensity, verticalBlur,horizontalBlur, borderSize):
        self.lumaCompressionRate = lumaCompressionRate
        self.lumaNoiseSigma = lumaNoiseSigma
        self.lumaNoiseMean = lumaNoiseMean
        self.chromaCompressionRate = chromaCompressionRate
        self.chromaNoiseIntensity = chromaNoiseIntensity
        self.verticalBlur = verticalBlur
        self.horizontalBlur = horizontalBlur
        self.borderSize = borderSize / 100
        self.generation = 3

    def addNoise(self, image, mean=0, sigma=30):
        height, width, channels = image.shape
        noisy_image = np.copy(image)
        gaussian_noise = np.random.normal(
            mean, sigma, (height, width, channels))
        noisy_image = np.clip(noisy_image + gaussian_noise,
                              0, 255).astype(np.uint8)
        return noisy_image

    def addChromaNoise(self, image, intensity=10):
        height, width = image.shape[:2]
        noise_red = np.random.randint(-intensity,
                                      intensity, (height, width), dtype=np.int16)
        noise_green = np.random.randint(-intensity,
                                        intensity, (height, width), dtype=np.int16)
        noise_blue = np.random.randint(-intensity,
                                       intensity, (height, width), dtype=np.int16)
        image[:, :, 0] = np.clip(image[:, :, 0] + noise_blue, 0, 255)
        image[:, :, 1] = np.clip(image[:, :, 1] + noise_green, 0, 255)
        image[:, :, 2] = np.clip(image[:, :, 2] + noise_red, 0, 255)
        image = np.uint8(image)
        return image

    def adjust_saturation(self,image, saturation_factor=1):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
        hsv_image[:, :, 1] = np.clip(
            hsv_image[:, :, 1] * saturation_factor, 0, 255)  # Adjust saturation channel
        output_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # Convert back to BGR
        return output_image

    def cut_black_line_border(self, image: np.ndarray, bordersize: int = None) -> None:
        h, w, _ = image.shape
        if bordersize is None:
            line_width = int(w * self.borderSize)  # 1.7%
        else:
            line_width = bordersize
        image[:, -1 * line_width:] = 0

    def compressLuma(self, image):
        height, width = image.shape[:2]
        step1 = cv2.resize(image, (int(width / self.lumaCompressionRate), int(height)),
                           interpolation=cv2.INTER_LANCZOS4)
        step1 = self.addNoise(step1, self.lumaNoiseMean, self.lumaNoiseSigma)
        
        step2 = cv2.resize(step1, (width, height),
                           interpolation=cv2.INTER_LANCZOS4)
        self.cut_black_line_border(step2)
        return step2
    def ringing(self,image, alpha, noiseSize, noiseValue):
        # Convert the input image to float32
        image_float32 = image.astype(np.float32)

        # Apply the DFT (Discrete Fourier Transform)
        dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

        # Generate ringing effect in the frequency domain
        rows, cols = image.shape[:2]
        center_row, center_col = rows // 2, cols // 2
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
                ringing_value = alpha * np.exp(-distance / noiseSize)
                dft[i, j] *= (1 + noiseValue * ringing_value)

        # Apply the inverse DFT to get the modified image
        modified_image = cv2.idft(dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # Convert the modified image back to uint8
        modified_image_uint8 = modified_image.astype(np.uint8)

        return modified_image_uint8
    def compressChroma(self, image):
        height, width = image.shape[:2]
        step1 = cv2.resize(image, (int(width / self.chromaCompressionRate), int(height)),
                           interpolation=cv2.INTER_LANCZOS4)
        step1 = self.addChromaNoise(step1, self.chromaNoiseIntensity)
        step2 = cv2.resize(step1, (width, height),
                           interpolation=cv2.INTER_LANCZOS4)
        self.cut_black_line_border(step2)
        return step2
    def blur(self, image):
        filtered_image = cv2.blur(image, (self.horizontalBlur,self.verticalBlur))
        return filtered_image
    
    def waves(self, img):
        rows, cols = img.shape[:2]

        # Create a meshgrid of indices
        i, j = np.indices((rows, cols))
        waves = round(random.uniform(0.000, 1.110), 3) * 1
        # Calculate the offset for each pixel
        offset_x = (waves * np.sin(250 * 2 * np.pi * i / (2 * cols))).astype(int)
        offset_j = j + offset_x

        # Clip the offset_j values to stay within the bounds of the image
        offset_j = np.clip(offset_j, 0, cols - 1)

        # Use advanced indexing to create the output image
        img_output = img[i, offset_j]

        return img_output
    def waves2(self, img):
        rows, cols = img.shape[:2]

        # Create a meshgrid of indices
        i, j = np.indices((rows, cols))
        waves = round(random.uniform(1.000, 1.110), 3) * 1
        # Calculate the offset for each pixel
        offset_x = ((waves * np.sin(np.cos(random.randint(200,250)) * 2 * np.pi * i / (2 * cols)))).astype(int)
        offset_j = j + offset_x

        # Clip the offset_j values to stay within the bounds of the image
        offset_j = np.clip(offset_j, 0, cols - 1)
        # Use advanced indexing to create the output image
        img_output = img[i, offset_j]

        return img_output
    def switchNoise(self, img):
        rows, cols = img.shape[:2]
        i, j = np.indices((rows, cols))
        waves = round(random.uniform(1.900, 1.910), 3)*1
        # Calculate the offset for each pixel
        offset_x = (waves * np.sin(np.cos(250) * 2 * np.pi * i / (2 * cols))).astype(int)
        offset_j = j + (offset_x*random.randint(20,30))

        # Clip the offset_j values to stay within the bounds of the image
        offset_j = np.clip(offset_j, 0, cols - 1)

        # Use advanced indexing to create the output image
        img_output = img[i, offset_j]
        return img_output
    def sharpen2(self,image, kernel_size=(5, 5), sigma=100, alpha=1.5, beta=-0.5):
        # Step 1: Apply Gaussian blur to the image
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)

        # Step 2: Calculate the unsharp mask
        unsharp_mask = cv2.addWeighted(image, alpha, blurred, beta, 0)

        # Ensure that pixel values are in the range [0, 255]
        unsharp_mask = np.clip(unsharp_mask, 0, 255)

        # Convert to uint8 (8-bit) image
        unsharp_mask = unsharp_mask.astype(np.uint8)

        return unsharp_mask
    def processFrame(self, image):
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        luma_compressed = self.compressLuma(image_ycrcb)
        chroma_compressed = self.compressChroma(image_ycrcb)
        chroma_compressed = self.waves(chroma_compressed)
        chroma_compressed = self.waves(chroma_compressed)
        chroma_compressed = cv2.medianBlur(chroma_compressed, 1)
        chrominance_layer = chroma_compressed[:, :, 1:3]
        merged_ycrcb = cv2.merge([luma_compressed[:, :, 0], chrominance_layer])
        chrominance_bgr = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)
        height, width, _ = chrominance_bgr.shape
        stripe_width = int(width * self.borderSize)
        chrominance_bgr[:, -stripe_width:, 1] = 0  # Set the green channel to 0 in the specified region
        return chrominance_bgr
    def sharpen_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        hpf = image - cv2.GaussianBlur(image, (21, 21), 3) + 127
        
        # Normalize images to the range [0, 1]
        image_norm = image.astype(float) / 255.0
        hpf_norm = hpf.astype(float) / 255.0
        
        # Soft light blending
        blended_norm = np.where(hpf_norm <= 0.5, 2 * image_norm * hpf_norm, 1 - 2 * (1 - image_norm) * (1 - hpf_norm))
        
        # Clip values to [0, 1]
        blended_norm = np.clip(blended_norm, 0, 1)
        
        # Convert back to standard image format
        blended = (blended_norm * 255).astype(np.uint8)
        
        return blended
    def applyRinging(self, yiq: np.ndarray, field: int):
        Y, I, Q = yiq
        sz = 0.5
        amp = 2
        shift = 0
        self._ringing = 0.5
        Y[field::2] = self.ringing(Y[field::2], self._ringing, clip=False)
        I[field::2] = self.ringing(I[field::2], self._ringing, clip=False)
        Q[field::2] = self.ringing(Q[field::2], self._ringing, clip=False)
        return yiq
    def bgr2yiq(self,bgrimg: np.ndarray) -> np.ndarray:
        planar = np.transpose(bgrimg, (2, 0, 1))
        b, g, r = planar
        dY = 0.30 * r + 0.59 * g + 0.11 * b

        Y = (dY * 256).astype(np.int32)
        I = (256 * (-0.27 * (b - dY) + 0.74 * (r - dY))).astype(np.int32)
        Q = (256 * (0.41 * (b - dY) + 0.48 * (r - dY))).astype(np.int32)
        return np.stack([Y, I, Q], axis=0).astype(np.int32)
    def yiq2bgr(self,yiq: np.ndarray, dst_bgr: np.ndarray = None, field: int = 0) -> np.ndarray:
        c, h, w = yiq.shape
        dst_bgr = dst_bgr if dst_bgr is not None else np.zeros((h, w, c))
        Y, I, Q = yiq
        if field == 0:
            Y, I, Q = Y[::2], I[::2], Q[::2]
        else:
            Y, I, Q = Y[1::2], I[1::2], Q[1::2]

        r = ((1.000 * Y + 0.956 * I + 0.621 * Q) / 256).astype(np.int32)
        g = ((1.000 * Y + -0.272 * I + -0.647 * Q) / 256).astype(np.int32)
        b = ((1.000 * Y + -1.106 * I + 1.703 * Q) / 256).astype(np.int32)
        r = np.clip(r, 0, 255)
        g = np.clip(g, 0, 255)
        b = np.clip(b, 0, 255)
        planarBGR = np.stack([b, g, r])
        interleavedBGR = np.transpose(planarBGR, (1, 2, 0))
        if field == 0:
            dst_bgr[::2] = interleavedBGR
        else:
            dst_bgr[1::2] = interleavedBGR
        return dst_bgr

    def applyVHSEffect(self, image):
        originalValues = [self.lumaNoiseSigma,self.lumaNoiseMean,self.chromaNoiseIntensity]
        self.lumaNoiseSigma = self.lumaNoiseSigma*self.generation
        self.lumaNoiseMean = self.lumaNoiseMean*self.generation
        self.chromaNoiseIntensity = self.chromaNoiseIntensity*self.generation
        image = self.processAll(image)
        self.lumaNoiseSigma = originalValues[0]
        self.lumaNoiseMean = originalValues[1]
        self.chromaNoiseIntensity = originalValues[2]
        return image
    
    def processAll(self,image):
        image = self.sharpen_image(image)  
        image = self.switchNoise(image)
        image = self.processFrame(image)
        image = self.waves(image)
        image = self.waves2(image)
        image = self.blur(image)
        image = self.sharpen2(image)
        return image
