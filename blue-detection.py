import cv2
import numpy as np
import os
import time

class BlueObjectDetector:
    def __init__(self):
        #rospy.init_node('blue_object_detector', anonymous=True)
        
        #self.bridge = CvBridge()
        #self.last_publish_time = rospy.Time.now()
        
        self.cap = cv2.VideoCapture(0)

        # Faixa de cores HSV para azul definido
        self.lower = np.array([90, 120, 70])
        self.upper = np.array([130, 255, 255])
        
        
        # Parâmetros geométricos
        self.min_area = 1000       # Área mínima em pixels
        self.max_area = 50000      # Área máxima em pixels
        self.max_boxes = 2         # Número máximo de bounding boxes
        
        # Subscriber e Publishers
        #self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        #self.image_pub = rospy.Publisher("/blue_object/detected_image", Image, queue_size=1)
        #self.coords_pub = rospy.Publisher("/blue_object/coordinates", Point, queue_size=1)
        
        os.system('cls')
        print("Detector de objetos azuis inicializado")

        ret, frame = self.cap.read()

        while ret:
            _, frame = self.cap.read()
            self.image_callback(msg=frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
            
        cv2.destroyAllWindows()
            
    def filter_contours(self, contours):
        """Filtra contornos por área e limita o número máximo"""
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                filtered.append(cnt)
        
        # Ordena por área (maior primeiro) e pega no máximo max_boxes
        filtered.sort(key=cv2.contourArea, reverse=True)
        return filtered[:self.max_boxes]

    def image_callback(self, msg):
        hsv = cv2.cvtColor(msg, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower, self.upper)
        cv2.imshow("mask", mask)
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        output_image = msg.copy()
        object_coords = None
        
        # Filtra contornos
        valid_contours = self.filter_contours(contours)
        
        # Processa no máximo max_boxes objetos
        for cnt in valid_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(output_image, (x,y), (x+w,y+h), (255,0,0), 3)
            center_x = x + w//2
            center_y = y + h//2
            cv2.putText(output_image, "AZUL", (x, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            
            # Usa as coordenadas do maior objeto
            if object_coords is None:
                object_coords = (center_x, center_y)
        
        self.publish_results(output_image)

    def publish_results(self, output_img):
        try:
            cv2.imshow("output", output_img)
          #  ros_img = self.bridge.cv2_to_imgmsg(output_img, "bgr8")
          #  ros_img.header = header
          #  self.image_pub.publish(ros_img)
            
          #  current_time = time.time()
          #  if coords and (current_time - self.last_publish_time).to_sec() >= 0.5:
          #      coord_msg = Point()
         #       coord_msg.x = coords[0]
         #       coord_msg.y = coords[1]
         #       coord_msg.z = 0.0
         #       self.coords_pub.publish(coord_msg)
         #       rospy.loginfo("Objeto azul detectado: X=%d, Y=%d", coords[0], coords[1])
           #     self.last_publish_time = current_time
        except:
            print('erro em publicar os resultados')
    #    except CvBridgeError as e:
      #      rospy.logerr("Erro ao publicar resultados: %s", e)

if __name__ == '__main__':
    #try:
  detector = BlueObjectDetector()
       # rospy.spin()
  #  except rospy.ROSInterruptException:
    #    pass
