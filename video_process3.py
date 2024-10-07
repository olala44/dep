from ultralytics import YOLO,solutions
import cv2
import threading
import smtplib
from email.message import EmailMessage
from email.mime.image import MIMEImage
from pathlib import Path

import smtplib
from email.message import EmailMessage
import numpy as np
from PIL import Image
import io
import threading

def email_alert(subject, body, to, image_array=None):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['To'] = to

    user = "hitmankillr@gmail.com"
    msg['From'] = user
    password = "yfatvcpvunibslmb"

    # Convert numpy image array to byte data and attach as image if provided
    print("i")
    if image_array is not None:
        print("ii")
        # Convert the numpy array to an image using Pillow
        img = Image.fromarray(image_array)
        
        # Create an in-memory bytes buffer to hold the image
        img_byte_arr = io.BytesIO()
        
        # Save the image to the in-memory buffer as a PNG (or any other format you prefer)
        img.save(img_byte_arr, format='PNG')
        
        # Get the byte data
        img_byte_arr = img_byte_arr.getvalue()
        
        # Attach the image to the email
        msg.add_attachment(img_byte_arr, maintype='image', subtype='png', filename='image.png')
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        print("done")

# Example usage



class Mo:
    def __init__(self,f):
        self.hash_id={}
        self.sp_time={}
        self.overlap=[]
        self.sent=[]
        self.time_limt=5
        self.model=YOLO("yolo11l.pt")
        self.model2=YOLO("best.pt")
        self.f=f
        self.is_acc=0
        self.is_car_stop=0

       
    def process(self,frame):
        
        time_limt=10
        acc_timelimt=10
        percent=int(frame.shape[0]*0.02)
        
        
        
        t=self.model.track(frame, persist=True, show=False,verbose=False,classes=[2,3,5,7],conf=0.2)
        acc_model=self.model2(frame, show=False,verbose=False,conf=0.2)

        img=t[0].plot(font_size=2,line_width=2)  

        temp,car_len= self.check_cars(t,acc_model)
        if(car_len>1):
            self.overlap+=temp
            self.overlap=list(set(self.overlap))
            img=self.add_for_acc(acc_model,img)
        try:  
            ids=(t[0].boxes.id.tolist())
            for index,i in enumerate(ids):
                if(i in self.hash_id):
                    self.hash_id[i]+=1
                else:
                    self.hash_id[i]=1

                if(i in self.overlap):

                    if(i in self.sp_time):
                        self.sp_time[i]+=1
                    else:
                        self.sp_time[i]=1
                    if(self.sp_time[i]/self.f>=acc_timelimt and i not in self.sent and self.is_acc!=1):
                        self.sent.append(i)
                        self.is_acc=1
                        threading.Thread(target=email_alert, args=("accident", "accident detected", "mody.a.d@hotmail.com", img)).start()
                if(self.is_acc==1):
                    img[0:percent,0:,:]=(255,0,0)

                    
            
                if(self.hash_id[i]/self.f>=time_limt):
                    img=cv2.putText(img,str(int(self.hash_id[i]/self.f)), (int(t[0].boxes.xywh[index][0].item()),int(t[0].boxes.xywh[index][1].item())), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
                    img[-percent:,0:,:]=(0,0,255)
                    self.is_car_stop=1
                    
                else:
                    img=cv2.putText(img,str(int(self.hash_id[i]/self.f)), (int(t[0].boxes.xywh[index][0].item()),int(t[0].boxes.xywh[index][1].item())), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,140),2)
                
        except:
            print("no object detected")
        return img,self.is_acc,self.is_car_stop

  


    def add_for_acc(self,pred,img):
        for i in pred[0]:
            x1, y1, x2, y2 = (i.boxes.xyxy[0][0].item(), 
                            i.boxes.xyxy[0][1].item(), 
                            i.boxes.xyxy[0][2].item(), 
                            i.boxes.xyxy[0][3].item())
            box_color = (0, 255, 0)  
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)        
            class_name = "accident"  
            confidence = i.boxes.conf[0].item()  
            confidence_text = f"{confidence:.2f}"
            
            text = f"{class_name} {confidence_text}"
            text_position = (int(x1), int(y1) - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            background_rect = (text_position[0], text_position[1] - text_height - baseline, 
                            text_width, text_height + baseline)

            cv2.rectangle(img, 
                        (background_rect[0], background_rect[1]), 
                        (background_rect[0] + background_rect[2], background_rect[1] + background_rect[3]), 
                        box_color,  
                        thickness=cv2.FILLED)  
            text_color = (255, 0, 255)  
            
        
            cv2.putText(img, text, text_position, font, font_scale, text_color, thickness)

        return img
    def check_cars(self,model,acc_model):
        
        if(len(acc_model[0].boxes.xyxy)==0):
            return 0,0
        try:
            reference_box=acc_model[0].boxes.xyxy[0]
            intersecting_boxes_indices = []
            for i in range( len(model[0].boxes.xyxy)):
                current_box = model[0].boxes.xyxy[i]
                if self.boxes_intersect(reference_box, current_box):
                    intersecting_boxes_indices.append(int(model[0].boxes.id[i]))
            return intersecting_boxes_indices,len(intersecting_boxes_indices)
        except:
            return 0,0
    
        
    def boxes_intersect(self,box1, box2):
        return not (box1[0] >= box2[2] or  # box1 is to the right of box2
                    box1[2] <= box2[0] or  # box1 is to the left of box2
                    box1[1] >= box2[3] or  # box1 is above box2
                    box1[3] <= box2[1])    # box1 is below box2


    

    

  