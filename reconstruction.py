import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import json

class StereoVisionProcessor:
    def __init__(self, calibration_file, left_image_file, right_image_file):
        with open(calibration_file, 'r') as f:
            self.calibration = json.load(f)

        self.imgL = cv2.imread(left_image_file)
        self.imgR = cv2.imread(right_image_file)
        self.imgLgray = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY)
        self.imgRgray = cv2.cvtColor(self.imgR, cv2.COLOR_BGR2GRAY)

    def compute_disparity_map(self):
        '''computing diaprity map with left and right image'''
        win_size = 3
        block_size = 7
        min_disp = 20
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=self.calibration['ndisp'],
            blockSize=block_size,
            uniquenessRatio=10,
            speckleWindowSize=3,
            speckleRange=1,
            P1=8 * 3 * win_size ** 2,
            P2=32 * 3 * win_size ** 2,
        )
        disparity_map = stereo.compute(self.imgLgray, self.imgRgray)
        disparity_map = np.float32(np.divide(disparity_map, 16.0))
        
        return disparity_map

    def plot_disparity_map(self, disparity_map):
        '''plotting disparity map compared to original image'''
        nrows, ncols = (1, 2)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
        axes[0].imshow(disparity_map, 'gray')
        axes[0].set_title('Disparity Map')
        axes[1].imshow(self.imgRgray, 'gray')
        axes[1].set_title('Original Image')
        plt.show()

    def rectify_images(self):
        '''rectification transforms, returns disparity-to-depth mapping matrix'''

        Q = np.zeros((4, 4))
        cv2.stereoRectify(
            cameraMatrix1=np.float64(self.calibration['cam0']),
            cameraMatrix2=np.float64(self.calibration['cam1']),
            distCoeffs1=0,
            distCoeffs2=0,
            imageSize=self.imgLgray.shape[:2],
            R=np.identity(3),
            T=np.array([0.54, 0., 0.]),
            R1=None,
            R2=None,
            P1=None,
            P2=None,
            Q=Q,
        )

        return Q
    
    def project_to_3D(self, disparity_map, Q):
        '''projecting disparity map using disparity to depth mapping matrix'''

        points_3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
        colours = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2RGB)
        mask_map = self.compute_disparity_map() > self.compute_disparity_map().min()
        output_points = points_3D[mask_map]
        output_colours = colours[mask_map]
        
        return output_points, output_colours

    def create_point_cloud_file(self, output_points, output_colours, output_file):
        '''creating ply files'''

        colours = output_colours.reshape(-1, 3)
        vertices = np.hstack([output_points.reshape(-1, 3), colours])

        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            '''
        with open(output_file, 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')

    def visualize_point_cloud(self, point_cloud_file):
        cloud = o3d.io.read_point_cloud(point_cloud_file)
        o3d.visualization.draw_geometries([cloud], window_name='Middlebury Art Reconstruction')

    def process(self, output_point_cloud_file):
        disparity_map = self.compute_disparity_map()
        Q = self.rectify_images()
        output_points, output_colours = self.project_to_3D(disparity_map, Q)
        self.plot_disparity_map(disparity_map)
        self.create_point_cloud_file( output_points, output_colours, output_point_cloud_file)
        self.visualize_point_cloud(output_point_cloud_file)


if __name__ == "__main__":
    calibration_file = './images/calib.json'
    left_image_file = './images/view1.png'
    right_image_file = './images/view5.png'
    output_point_cloud_file = 'pointCloud.ply'

    processor = StereoVisionProcessor(calibration_file, left_image_file, right_image_file)
    processor.process(output_point_cloud_file)
