'''
@Author: Qi Er Teng
@GitHub: dwz92
@Discription: This script includes the python GUI to run multiple training process for designated models
'''



#### WORKPLACE SETUP ####
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import AOAMLP as au
import multiprocessing
import test
##########################



# Global Var
WIDTH = 1000
HEIGHT = 500

ENTRIES = []

display=[]

# Canvas Set Up
root = tk.Tk(screenName="Main", baseName="Multiprocess Model Run")
root.geometry(str(WIDTH) + 'x' + str(HEIGHT))
root.resizable(0, 0)
root.title('Multiprocess Model Run')
root.config(background="#FFFFFF")


def parser(d):

    freq1 = d['freq_from']
    freq2 = d['freq_to']
    freq = freq1 +'_' + freq2

    snr = int(d['SNR'])

    saveto = d['saveto']

    srcdir = d['Src_Dir']

    k = int(d['K'])

    hiddenr = d['Hidden'].split(',')
    hidden = list(map(int, hiddenr))

    refchan = int(d['ref'])

    out = int(d['Output'])

    device = d['Device']

    parsed = (freq, snr, saveto, srcdir, k, hidden, refchan, out, device)

    return parsed


def runAll():
    maxrun = int(spin1.get())
    print(maxrun)
    args = []

    for dict in display:
        command = parser(dict)
        print(command)
        args.append(command)

    with multiprocessing.Pool(processes=maxrun) as pool:
        try:
            pool.starmap(au.runTrain, args)
            print("All processes completed successfully")
        except Exception as e:
            print(f"Error occurred: {e}")
            pool.terminate()
            pool.close()
        finally:
            pool.close()
            pool.join()

    print(f'Multiprocess training session finished.')


def pauseAll():
    for entry in ENTRIES:
        entry.config(state='disabled')

    runAll()


def update_display():
    # Clear old labels if needed
    for widget in display_frame.winfo_children():
        widget.destroy()
    
    for line in display:
        # Display key info or format as needed
        # text = f"{line['Hidden']} | {line['K']} | {line['Output']}"
        newlab = tk.Label(display_frame, text=line, anchor='w', bg="#9AD3C2")
        newlab.pack(anchor='w', padx=5, pady=2)




def newPro():
    def browseSrcDir():
        dir = filedialog.askdirectory()
        list2.delete(0, tk.END)
        if dir:
            list2.insert(tk.END, dir)
    
    def browseSaveto():
        dir = filedialog.askdirectory()
        list1.delete(0, tk.END)
        if dir:
            list1.insert(tk.END, dir)
    
    def completed():
        result = {
            'Hidden': entry4.get(),
            'K': entry5.get(),
            'Output': entry6.get(),
            'freq_from':entry7.get(),
            'freq_to': entry8.get(),
            'ref': entry9.get(),
            'SNR': entry10.get(),
            'Device': entry3.get(),
            'Src_Dir': list2.get(0) if list2.size() > 0 else '',
            'saveto': list1.get(0) if list1.size() > 0 else ''
        }

        display.append(result)
        update_display()
        win.destroy()

        
    # Global Var
    WIDTH2 = 500
    HEIGHT2 = 500

    # Canvas Set Up
    win = tk.Tk(screenName="New Process", baseName="Multiprocess Model Run")
    win.geometry(str(WIDTH2) + 'x' + str(HEIGHT2))
    win.resizable(0, 0)
    win.title('New Process')
    win.config(background="#C7CACA")

    
    ##MODEL INFO##
    # Hidden
    label5 = tk.Label(win, text="Hidden Layers Sizes (e.g. 12,36,125): ")
    label5.place(x=WIDTH2*0.1, y=HEIGHT2*0.05)
    entry4 = tk.Entry(win, width=60)
    entry4.place(x=WIDTH2*0.1, y=HEIGHT2*0.05+20)
    # Ele
    label6 = tk.Label(win, text="Number of Elements in Antenna Array: ")
    label6.place(x=WIDTH2*0.1, y=HEIGHT2*0.18)
    entry5 = tk.Entry(win, width=10)
    entry5.place(x=WIDTH2*0.5+12, y=HEIGHT2*0.18)
    # output 
    label7 = tk.Label(win, text="Output: ")
    label7.place(x=WIDTH2*0.7, y=HEIGHT2*0.18)
    entry6 = tk.Entry(win, width=10)
    entry6.place(x=WIDTH2*0.8, y=HEIGHT2*0.18)


    ##DATASET INFO##
    # freq
    label8 = tk.Label(win, text="Frequency band: ")
    label8.place(x=WIDTH2*0.1, y=HEIGHT2*0.3)
    entry7 = tk.Entry(win, width=5)
    entry7.place(x=WIDTH2*0.3, y=HEIGHT2*0.3)
    label8_2 = tk.Label(win, text="to")
    label8_2.place(x=WIDTH2*0.37, y=HEIGHT2*0.3)
    entry8 = tk.Entry(win, width=5)
    entry8.place(x=WIDTH2*0.4, y=HEIGHT2*0.3)
    # ref
    label9 = tk.Label(win, text="Reference Channel: ")
    label9.place(x=WIDTH2*0.5, y=HEIGHT2*0.3)
    entry9 = tk.Entry(win, width=5)
    entry9.place(x=WIDTH2*0.72, y=HEIGHT2*0.3)
    # SNR
    label10 = tk.Label(win, text="Signal to Noise Ratio (SNR): ")
    label10.place(x=WIDTH2*0.1, y=HEIGHT2*0.4)
    entry10 = tk.Entry(win, width=5)
    entry10.place(x=WIDTH2*0.4, y=HEIGHT2*0.4)


    ##TRAINING INFO##
    # Device
    label4 = tk.Label(win, text="Device (e.g. cuda):")
    label4.place(x=WIDTH2*0.1, y=HEIGHT2*0.5)
    entry3 = tk.Entry(win, width=5)
    entry3.place(x=WIDTH2*0.3+5, y=HEIGHT2*0.5)
    # src_dir
    label2 = tk.Label(win, text="Source Directory:")
    label2.place(x=WIDTH2*0.1, y=HEIGHT2*0.6)
    button2 = tk.Button(win, text="Browse Folder...", command=browseSrcDir)
    button2.place(x=WIDTH2*0.1, y=HEIGHT2*0.65)
    list2 = tk.Listbox(win, selectmode=tk.SINGLE, height=1, width=50)
    list2.place(x=WIDTH2*0.1+100, y=HEIGHT2*0.6)
    # save_to
    label3 = tk.Label(win, text="Save Model To:")
    label3.place(x=WIDTH2*0.1, y=HEIGHT2*0.75)
    button3 = tk.Button(win, text="Browse Folder...", command=browseSaveto)
    button3.place(x=WIDTH2*0.1, y=HEIGHT2*0.8)
    list1 = tk.Listbox(win, selectmode=tk.SINGLE, height=1, width=50)
    list1.place(x=WIDTH2*0.1+100, y=HEIGHT2*0.75)


    #Complete
    button4 = tk.Button(win, text="Complete", bg="#57E78E", command=completed)
    button4.place(x=WIDTH2*0.8, y=WIDTH2*0.85)




# Delete Process
def delPro():
    # GLOBAL VAR
    WIDTH3 = 500
    HEIGHT3 = 500

    def delete():
        selected = drop.curselection()

        if not selected:
            return

        del display[selected[0]]
        update_display()
        win2.destroy()

    win2 = tk.Tk(screenName="Delete Process", baseName="Multiprocess Model Run")
    win2.geometry(str(WIDTH3) + 'x' + str(HEIGHT3))
    win2.resizable(0, 0)
    win2.title('Delete Process')
    win2.config(background="#C7CACA")

    drop = tk.Listbox(win2, selectmode=tk.SINGLE, height=1, width=50)
    for pro in display:
        drop.insert(tk.END, pro)
    drop.place(x=WIDTH3*0.2, y=0, height=HEIGHT3)

    compbutton = tk.Button(win2, text="Complete", bg="#57E78E", command=delete)
    compbutton.place(x=WIDTH3*0.83, y=HEIGHT3*0.85)



def newPre():
    # GLOBAL VAR
    WIDTH4 = 500
    HEIGHT4 = 500

    
    def browseSrcFile():
        dir = filedialog.askopenfilename()
        list2.delete(0, tk.END)
        if dir:
            list2.insert(tk.END, dir)

    def browseModelFile():
        dir = filedialog.askopenfilename()
        list4.delete(0, tk.END)
        if dir:
            list4.insert(tk.END, dir)
    
    def browseSaveto():
        dir = filedialog.askdirectory()
        list1.delete(0, tk.END)
        if dir:
            list1.insert(tk.END, dir)

    def runPredict():
        mod=list3.get()
        sourcedata = list2.get(0) if list2.size() > 0 else ''
        modelpath = list4.get(0) if list4.size() > 0 else ''
        savetodir = list1.get(0) if list1.size() > 0 else ''

        if mod == 'Prediction':
            test.pred(modelfile=modelpath, testdata=sourcedata, saveto=savetodir, header=False, device='cuda')
        else:
            test.test(modelpath, sourcedata, savetodir, 'cuda')

        win3.destroy()

    win3 = tk.Tk(screenName="Prediction", baseName="Multiprocess Model Run")
    win3.geometry(str(WIDTH4) + 'x' + str(HEIGHT4))
    win3.resizable(0, 0)
    win3.title('Prediction')
    win3.config(background="#C7CACA")

    # Mode
    label4 = tk.Label(win3, text="Mode: ")
    label4.place(x=WIDTH4*0.1, y=HEIGHT4*0.2)
    modes = ['Prediction', 'Testing']
    list3 = ttk.Combobox(win3, values=modes,state='readonly', height=1, width=20)
    list3.place(x=WIDTH4*0.1+50, y=HEIGHT4*0.2)
    list3.set("Select a mode")

    # src_data
    label2 = tk.Label(win3, text="Source Data File:")
    label2.place(x=WIDTH4*0.1, y=HEIGHT4*0.4)
    button2 = tk.Button(win3, text="Browse File...", command=browseSrcFile)
    button2.place(x=WIDTH4*0.1, y=HEIGHT4*0.45)
    list2 = tk.Listbox(win3, selectmode=tk.SINGLE, height=1, width=50)
    list2.place(x=WIDTH4*0.1+100, y=HEIGHT4*0.4)
    # src_model
    label4 = tk.Label(win3, text="Source Model File:")
    label4.place(x=WIDTH4*0.1, y=HEIGHT4*0.55)
    button4 = tk.Button(win3, text="Browse File...", command=browseModelFile)
    button4.place(x=WIDTH4*0.1, y=HEIGHT4*0.6)
    list4 = tk.Listbox(win3, selectmode=tk.SINGLE, height=1, width=50)
    list4.place(x=WIDTH4*0.1+100, y=HEIGHT4*0.55)
    # save_to
    label3 = tk.Label(win3, text="Save Result To:")
    label3.place(x=WIDTH4*0.1, y=HEIGHT4*0.7)
    button3 = tk.Button(win3, text="Browse Folder...", command=browseSaveto)
    button3.place(x=WIDTH4*0.1, y=HEIGHT4*0.75)
    list1 = tk.Listbox(win3, selectmode=tk.SINGLE, height=1, width=50)
    list1.place(x=WIDTH4*0.1+100, y=HEIGHT4*0.7)

    # Predict Button
    button2 = tk.Button(win3, text="Complete", bg="#57E78E", command=runPredict)
    button2.place(x=WIDTH4*0.1, y=HEIGHT4*0.9)






# Menu
menubar = tk.Menu(root, background="#9AA09C")

# Process Menu
promenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Training Progress', menu=promenu)
promenu.add_command(label ='New Process', command = newPro)
promenu.add_command(label='Delete Process', command=delPro)

# Predict
premenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Predictions', menu=premenu)
premenu.add_command(label ='New Predict', command = newPre)


# Help Menu
helpmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label ='About', command = None)
helpmenu.add_command(label='Contact')
helpmenu.add_separator()
helpmenu.add_command(label='Exit', command=root.destroy)



# Max process
label1 = tk.Label(root, text="Maximum Parallel Process:")
label1.place(x=WIDTH*0.82, y=HEIGHT*0.02)
spin1 = tk.Spinbox(root, from_=1, to=10, width=22)
spin1.place(x=WIDTH*0.82, y=HEIGHT*0.02+20)
ENTRIES.append(spin1)

# Start Button
button1 = tk.Button(root, text="Start", bg="#57E78E", command=pauseAll)
button1.place(x=WIDTH*0.02, y=HEIGHT*0.02)
ENTRIES.append(button1)

# displays
display_frame = tk.Frame(root, width=300, height=200, bg="#9AD3C2")
display_frame.place(x=WIDTH*0.1, y=0, width=700, height=HEIGHT)
display_frame.pack_propagate(False)

labels = []
for line in display:
    print(line)
    newlab = tk.Label(display_frame, text=line)
    newlab.pack(anchor='w')


# root.config(menu=menubar)
# root.mainloop()

if __name__ == "__main__":
    multiprocessing.freeze_support()  # for compatibility on Windows
    root.config(menu=menubar)
    root.mainloop()
