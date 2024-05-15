def gaussian_laplacian_kernel(size, sigma):
    """Genera un kernel de convolución para la laplaciana de Gaussiana."""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*sigma**4)) * ((i - center)**2 + (j - center)**2 - 2*sigma**2) * np.exp(-((i - center)**2 + (j - center)**2)/(2*sigma**2))
    return kernel

def convolve2d(image, kernel):
    """Realiza la convolución 2D."""
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    result = np.zeros((y,x))
    for i in range(y):
        for j in range(x):
            result[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return result

def smooth_image(image, sigma):
    """Suaviza una imagen utilizando un filtro Gaussiano."""
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_laplacian_kernel(kernel_size, sigma)
    smoothed_image = convolve2d(image, kernel)
    return smoothed_image