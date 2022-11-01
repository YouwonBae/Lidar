# =============================================
# represent laser data of 2d lidar
# =============================================
import numpy as np
import math
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Lidar represent laser data')
    parser.add_argument('--mode', default=0, type=int,
                        help='mid = 0 / mindist = 1 / minangle = 2')
    parser.add_argument('--hardware_midangle', default=90, type=int,
                        help='hardware_midangle')
    parser.add_argument('--software_midangle', default=180, type=int,
                        help='software_midangle')
    parser.add_argument('--start_angle', default=90, type=int,
                        help='start_angle')
    parser.add_argument('--end_angle', default=270, type=int,
                        help='end_angle')
    parser.add_argument('--maxdist', default=1.00, type=float,
                        help='maxdist')
    parser.add_argument('--mindist', default=0.05, type=float,
                        help='mindist')
    parser.add_argument('--linear_distance', default=0.06, type=int,
                        help='linear_distance')
    parser.add_argument('--linear_angle', default=4, type=int,
                        help='linear_angle')
    parser.add_argument('--obstacle_minsize', default=0.05, type=float,
                        help='obstacle_minsize')
    parser.add_argument('--obstacle_maxsize', default=1.00, type=float,
                        help='obstacle_maxsize')
    parser.add_argument('--raw_cloud', default=True, type=str2bool,
                        help='Whether show raw lidar data')
    parser.add_argument('--filtered_cloud', default=True, type=str2bool,
                        help='Whether show filtered lidar data')
    global args
    args = parser.parse_args(argv)

class Lidar():
    def __init__(self):
        self.distance = np.zeros(shape=(360), dtype=np.float64) #라이다 데이터를 담을 변수
        # =============================================
        # obstacleData에서 사용할 변수, 저장공간 선언부
        # =============================================
        # hardware angle
        self.hardware_midangle = args.hardware_midangle
        # lidar ROI setup
        self.mid_angle = args.software_midangle
        self.start_angle = args.start_angle
        self.end_angle = args.end_angle
        self.mindist = args.mindist
        self.maxdist = args.maxdist
        # obstacle filter setup
        self.linear_distance = args.linear_distance
        self.linear_angle = args.linear_angle
        self.obstacle_minsize = args.obstacle_minsize
        self.obstacle_maxsize = args.obstacle_maxsize

    def obstacleData(self, raw_distance):
        # lidar array
        pointcloud = []
        objcloud = np.empty((0,2), int)
        linear_classes = np.empty((0,3), int)
        object_data = np.empty((0,4), int)
        cloud_cnt = 1
        # lidar ROI setup
        mid_angle = self.mid_angle
        start_angle = self.start_angle
        end_angle = self.end_angle
        mindist = self.mindist
        maxdist = self.maxdist
        # obstacle filter setup
        linear_distance = self.linear_distance
        linear_angle = self.linear_angle
        obstacle_minsize = self.obstacle_minsize
        obstacle_maxsize = self.obstacle_maxsize

        angle_err = self.hardware_midangle - mid_angle

        for i in range(start_angle, end_angle):
            i += angle_err
            if i > 359: i -= 360
            elif i < 0: i += 360

            if mindist <= raw_distance[i] <= maxdist:
                pointcloud = np.append(pointcloud, [raw_distance[i]])
                objcloud = np.append(objcloud, [[raw_distance[i], i + start_angle]], axis = 0)
            elif raw_distance[i] > maxdist:
                pointcloud = np.append(pointcloud, [np.inf])
            else:
                pointcloud = np.append(pointcloud, [0])
            
        for i in range(len(objcloud) - 2):
            i += 1
            if abs(objcloud[i - 1, 0] - objcloud[i, 0]) < linear_distance and abs(objcloud[i - 1, 1] - objcloud[i, 1]) < linear_angle:
                linear_classes = np.append(linear_classes, [[ cloud_cnt, objcloud[i, 0], objcloud[i, 1] ]], axis = 0)
                if abs(objcloud[i, 0] - objcloud[i+1, 0]) < linear_distance and abs(objcloud[i, 1] - objcloud[i+1, 1]) < linear_angle: None
                else: cloud_cnt += 1

        if args.filtered_cloud:
            print('\nROIed data')
            print(pointcloud)
            print('\nobject data')
            print(objcloud)
            print('\nclassed data')
            print(linear_classes)

        for cnt in range(cloud_cnt):
            #initialize
            obstacle_size = 0
            min_angle = 360
            max_angle = 0
            for i, linear_class in enumerate(linear_classes): # check object's features
                if linear_classes[i, 0] == cnt + 1:
                    if i < len(linear_classes) - 1 and linear_classes[i + 1, 0] == cnt + 1:
                        l_now = linear_classes[i, 1]
                        l_next = linear_classes[i + 1, 1]
                        inc_angle = linear_classes[i + 1, 2] - linear_classes[i, 2]
                        obj_piece = math.sqrt((l_now * (math.sin(math.pi * (float(inc_angle) / 180)))) ** 2 
                                                + abs(l_next - l_now * (math.cos(math.pi * (float(inc_angle) / 180)))) ** 2)
                        obstacle_size += obj_piece
                    if linear_classes[i, 2] < min_angle:
                        min_angle = linear_classes[i, 2]
                        min_dist = linear_classes[i, 1]
                    if linear_classes[i, 2] > max_angle:
                        max_angle = linear_classes[i, 2]
                        max_dist = linear_classes[i, 1]

            # get object_data from a obstacle in linear class's
            if obstacle_minsize <= obstacle_size <=  obstacle_maxsize:
                obstacle_angle = np.mean(linear_classes[np.where(linear_classes[:, 0] == cnt + 1), 2])

                if args.mode == 0:
                    represent_angle = obstacle_angle
                    represent_dist = linear_classes[round(np.mean(np.where(linear_classes[:, 0] == cnt + 1))), 1]
                    
                elif args.mode == 1:
                    represent_dist = np.min(linear_classes[np.where(linear_classes[:, 0] == cnt + 1), 1])

                    if obstacle_angle > mid_angle:  # right obstacle
                        represent_angle = np.min(linear_classes[np.where(linear_classes[:, 0] == cnt + 1), 2])
                    else:                           # left obstacle
                        represent_angle = np.max(linear_classes[np.where(linear_classes[:, 0] == cnt + 1), 2])

                elif args.mode == 2:
                    if obstacle_angle > mid_angle:  # right obstacle
                        represent_dist = min_dist
                        represent_angle = min_angle
                    else:                           # left obstacle
                        represent_dist = max_dist
                        represent_angle = max_angle

                else: print('that is not valid mode')

                # calculate vertical distance (left: negative, right: positive)
                if represent_angle < mid_angle:
                    sin_angle = mid_angle - represent_angle
                    vertical_dist = -1 * represent_dist * (math.sin(math.pi * (float(sin_angle) / 180)))
                else:
                    sin_angle = represent_angle - mid_angle
                    vertical_dist = represent_dist * (math.sin(math.pi * (float(sin_angle) / 180)))
                object_data = np.append(object_data, [[ represent_dist*1000, represent_angle, int(vertical_dist*1000), obstacle_size*1000]], axis = 0 ) 

            else: None

        return object_data

    # =============================================
    # 메인 함수 
    # =============================================
    def start(self):

        mod_table = ['midle angle', 'minimum distance', 'minimum angle']
        print('\nmode: %s' %mod_table[args.mode])

        
        self.distance[30:50] = 1.0
        self.distance[50] = 0.98
        self.distance[51:100] = 1.0
        self.distance[120:200] = 0.5
        self.distance[250:350] = 0.3
        

        if args.raw_cloud:
            print('\nraw cloud')
            print(self.distance)

        # get obstacle data
        object_data = self.obstacleData(self.distance)
        print('\nresult[ distance, angle, vertical distance, obstacle size ]')
        print(object_data)


# =============================================
# 가장 먼저 호출되는 함수로 여기서 start() 함수를 호출함.
# =============================================
if __name__ == '__main__':
    parse_args()
    node = Lidar()
    node.start()