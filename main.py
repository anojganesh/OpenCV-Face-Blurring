import cv2

def analyze_video(video_path):
    success, frame = video_path.read()
    height = 0
    width = 0
    if success:
        height = frame.shape[0]
        width = frame.shape[1]
    return success, frame, width, height

def initialize_output(width, height):
    
    # create blank video to add frames to
    output = cv2.VideoWriter('assets/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    return output

def main():
    video = cv2.VideoCapture('assets/video1.mp4')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    success, frame, width, height = analyze_video(video)

    output = initialize_output(width, height)

    while success:
        # scale up 10%, 6 overlaps 
        faces = face_cascade.detectMultiScale(frame, 1.1, 6)
        for(x,y,w,h) in faces:
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255),9)

            # add blur to the rectangle box
            frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (50,50))

       

        output.write(frame)

        # get next frame
        success, frame, width, height = analyze_video(video)

    # clean up
    video.release()
    output.release()
    cv2.destroyAllWindows()

    # generating result takes ~20seconds
    print("Finished")

if __name__ == "__main__":
    main()


