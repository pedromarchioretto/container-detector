#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

class ContainerDetector:
    def __init__(self):
        rospy.init_node('container_detector', anonymous=True)
        
        self.bridge = CvBridge()
        self.last_publish_time = rospy.Time.now()
        
        # Faixas de cores HSV (ajuste conforme necessário)
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([100, 120, 70])
        self.blue_upper = np.array([130, 255, 255])
        
        # Parâmetros geométricos
        self.min_area = 1000  # Área mínima em pixels
        
        # Subscriber e Publishers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/container/detected_image", Image, queue_size=1)
        self.red_pub = rospy.Publisher("/container/red_coordinates", Point, queue_size=1)
        self.blue_pub = rospy.Publisher("/container/blue_coordinates", Point, queue_size=1)
        
        rospy.loginfo("Detector de containers inicializado")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detecção de vermelho (2 faixas HSV)
            red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Detecção de azul
            blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
            
            # Processamento morfológico
            kernel = np.ones((5,5), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            _, red_contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _, blue_contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            output_image = cv_image.copy()
            red_coords = None
            blue_coords = None
            
            # Processar containers vermelhos
            for cnt in red_contours:
                area = cv2.contourArea(cnt)
                if area > self.min_area:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(output_image, (x,y), (x+w,y+h), (0,0,255), 3)
                    center_x = x + w//2
                    center_y = y + h//2
                    cv2.putText(output_image, "VERMELHO", (x, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    red_coords = (center_x, center_y)
            
            # Processar containers azuis
            for cnt in blue_contours:
                area = cv2.contourArea(cnt)
                if area > self.min_area:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(output_image, (x,y), (x+w,y+h), (255,0,0), 3)
                    center_x = x + w//2
                    center_y = y + h//2
                    cv2.putText(output_image, "AZUL", (x, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    blue_coords = (center_x, center_y)
            
            # Publicar resultados
            self.publish_results(output_image, red_coords, blue_coords, msg.header)
            
        except CvBridgeError as e:
            rospy.logerr("Erro no processamento: %s", e)

    def publish_results(self, output_img, red_coords, blue_coords, header):
        try:
            # Publicar imagem com detecções
            ros_img = self.bridge.cv2_to_imgmsg(output_img, "bgr8")
            ros_img.header = header
            self.image_pub.publish(ros_img)
            
            # Publicar coordenadas
            current_time = rospy.Time.now()
            if (current_time - self.last_publish_time).to_sec() >= 1.0:
                if red_coords:
                    red_msg = Point()
                    red_msg.x = red_coords[0]
                    red_msg.y = red_coords[1]
                    red_msg.z = 0.0
                    self.red_pub.publish(red_msg)
                    rospy.loginfo("Container vermelho: X=%d, Y=%d", red_coords[0], red_coords[1])
                
                if blue_coords:
                    blue_msg = Point()
                    blue_msg.x = blue_coords[0]
                    blue_msg.y = blue_coords[1]
                    blue_msg.z = 0.0
                    self.blue_pub.publish(blue_msg)
                    rospy.loginfo("Container azul: X=%d, Y=%d", blue_coords[0], blue_coords[1])
                
                self.last_publish_time = current_time
                
        except CvBridgeError as e:
            rospy.logerr("Erro ao publicar resultados: %s", e)

if __name__ == '__main__':
    try:
        detector = ContainerDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass