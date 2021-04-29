#code created with help from a project created by the DataFlair Team - https://data-flair.training/blogs/python-chatbot-project/
#creating a GUI using Tkinter library - take input message from suer and use helper functions that have been created
import tkinter
from tkinter import *
from chatbotapp import chatbot_response

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#ffffff", font=("Arial", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bakerbot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Bakerbot - Ask me about the bakery!")
base.geometry("400x500")
base.resizable(width=TRUE, height=TRUE)

#create the chat window
ChatLog = Text(base, bd=0, bg="black", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#bind a scrollbar to the chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#create a button that allows the user to send a message to the bot
SendButton = Button(base, font=("Arial",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#dbb6f2", activebackground="#cc8cf5", fg='#ffffff', command= send)

#create box for a message to be entered 
EntryBox = Text(base, bd=0, bg="white", width="30", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

#put all components on the window
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
