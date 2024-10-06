from ultralytics import YOLO,solutions
import cv2




class Mo:
    def __init__(self,f):
        self.hash_id={}
        self.sp_time={}
        self.overlap=[]
        self.sent=[]
        self.time_limt=5
        self.model=YOLO("yolo11l.pt",)
        self.model2=YOLO("best.pt")
        self.f=f
        self.is_acc=0

       
    def process(self,frame):
        
        time_limt=30
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
                    if(self.sp_time[i]/self.f>=acc_timelimt and i not in self.sent):
                        self.sent.append(i)
                        self.is_acc=1
                    #email_alert("hey", "hello thabit", "mody.a.d@hotmail.com")
                if(self.is_acc==1):
                    img[0:percent,0:,:]=(255,0,0)

                    
            
                if(self.hash_id[i]/self.f>=time_limt):
                    img=cv2.putText(img,str(int(self.hash_id[i]/self.f)), (int(t[0].boxes.xywh[index][0].item()),int(t[0].boxes.xywh[index][1].item())), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)
                    img[-percent:,0:,:]=(0,0,255)
                    
                else:
                    img=cv2.putText(img,str(int(self.hash_id[i]/self.f)), (int(t[0].boxes.xywh[index][0].item()),int(t[0].boxes.xywh[index][1].item())), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,140),2)
                
        except:
            print("no object detected")
        return img

  


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
        reference_box=acc_model[0].boxes.xyxy[0]
        intersecting_boxes_indices = []
        for i in range( len(model[0].boxes.xyxy)):
            current_box = model[0].boxes.xyxy[i]
            if self.boxes_intersect(reference_box, current_box):
                intersecting_boxes_indices.append(int(model[0].boxes.id[i]))
        return intersecting_boxes_indices,len(intersecting_boxes_indices)

        
    def boxes_intersect(self,box1, box2):
        return not (box1[0] >= box2[2] or  # box1 is to the right of box2
                    box1[2] <= box2[0] or  # box1 is to the left of box2
                    box1[1] >= box2[3] or  # box1 is above box2
                    box1[3] <= box2[1])    # box1 is below box2


    

    

  