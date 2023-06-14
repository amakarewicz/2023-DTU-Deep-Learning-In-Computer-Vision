import numpy as np
import sys

image = str(sys.argv[1])
method = str(sys.argv[2])
step = float(sys.argv[3])

projection = np.load(f'../latent_codes/{image}/projected_w.npz')['w']
projection_transformed = projection.copy()
direction = np.load(f'../stylegan2directions/{method}.npy')

projection_transformed[0] = projection[0] + direction*step
np.savez(f'../latent_codes/{image}/{method}_{step}_projected_w.npz', w=projection_transformed)