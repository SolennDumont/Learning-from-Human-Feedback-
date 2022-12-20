import speech_recognition as sr



"""
Function purpose is to transcibe user feedback into text 
"""
def getText(): 
   
    r = sr.Recognizer() #Initialising the recognizeer
    
    mic = sr.Microphone() #setting 
    err = True #To get out of the loop
    with mic as source :
      print("Can please give your feedback")
      audio = r.adjust_for_ambient_noise(source) 
      audio = r.listen(source) #beginning to aquire the audio

    try:
       text = r.recognize_google(audio) #Transcribing
       print("Your feedback is :", text)
    except sr.UnknownValueError: #if we face an error while transcribing
      print("Cannot understand what you said")
    if text == 'good': #Catch
      f = 0 #Good feed back
    elif text == 'bad':
      f = 1 #Bad feed back
    elif text == 'next':
      f = 3 #Bad feed back (too far)
    elif text == 'neutral':
      f = 2
    else :
      f = 2 #no feedback
                  
    return f 




