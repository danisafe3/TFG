from tkinter import *
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image

import numpy as np
import nrrd
import time

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

cmd_messages = { "model_load":"---\n OPERATION: Model successfully loaded.\n---\n",
				 "weights_load":"---\n OPERATION: Weights successfully loaded.\n---\n",
				 "model_error":"---\n ERROR: Model couldn't be loaded.\n---\n",
				 "weights_error":"---\n ERROR: Weights couldn't be loaded.\n---\n",
				 "done_error":"---\n ERROR: No segmentation found.\n---\n",
				 "m_w_missing":"---\n ERROR: Either model or weights were not loaded.\n---\n",
				 "image_missing":"---\n ERROR: Image was not loaded.\n---",
 				 "size_error":"---\n ERROR: Image size does not match the expected input size of the model.\n---\n",
				 "order_error":"---\n ERROR: Attempted to load weights before architecture.\n---\n",
				 "file_error":"---\n File error\n---\n",
				 "other_error":"---\n OTHER ERROR\n---\n",
				 "stopped":"---\n OPERATION: User cancelled segmentation manually.\n---\n"
				 }


# Resets the status to "waiting"
def status_reset():
	status_label["text"]=status_waiting


# Sends error message and terminates the operation
def error_and_reset(cmd_text=cmd_messages["other_error"], msgbox_title="OTHER ERROR", msgbox_text="Some error happened."):
	messagebox.showerror(msgbox_title, msgbox_text)
	sys.stdout.write(cmd_text)
	status_reset()



def ok_and_reset(cmd_text, label, label_text):
	sys.stdout.write(cmd_text)
	label.config(text=label_text, fg='green')
	status_reset()



def check_or_reset():
	answer = messagebox.askokcancel(message="You are about to create the segmentation mask. This may take a few seconds. Do you want to continue?")
	if answer:
		return(answer)
	else:
		sys.stdout.write(cmd_messages["stopped"])
		status_reset()
		return(answer)



def cpu_switch():
	global cpu_flag
	if cpu_flag:
		cpu_flag = False
		cpu_btn["text"] = "Force CPU: OFF"
	else:
		cpu_flag = True
		cpu_btn["text"] = "Force CPU: ON"



def model_structure_loader():

	global loaded_model, model_structure_file, model_flag
	
	status_label["text"]=status_architecture

	model_structure_file = filedialog.askopenfilename(initialdir="/", title="Select Model Structure File", filetypes=(("json files", "*.json"),("all files","*.*")))
	if model_structure_file == '':
		error_and_reset(cmd_messages["file_error"],"File error","No file selected.")
		return(-1)

	json_file = open(model_structure_file, 'r')
	model_structure = json_file.read()
	json_file.close()

	if model_flag:
		sys.stdout.write("---\nWARNING: Model already loaded. Attempting to load new model.\n---\n")

	try:
		loaded_model = keras.models.model_from_json(model_structure)
		model_flag = True
		ok_and_reset(cmd_messages["model_load"], model_label, model_text)

	except:
		error_and_reset(cmd_messages['model_error'], "Model Error", "The model cannot be loaded correctly.")
		return(-1)



def model_weights_loader():

	global loaded_model, model_weights_file, weights_flag

	status_label["text"]=status_weights

	model_weights_file = filedialog.askopenfilename(initialdir="/", title="Select Model Weights File", filetypes=(("h5 files", "*.h5"),("all files","*.*")))
	if model_weights_file == '':
		error_and_reset(cmd_messages["file_error"],"File Error","No file selected.")
		return(-1)

	if model_flag:
		try:
			loaded_model.load_weights(model_weights_file)
			weights_flag = True
			ok_and_reset(cmd_messages["weights_load"], weights_label, weights_text)

		except:
			error_and_reset(cmd_messages["weights_error"], "Weights Error", "The weights cannot be loaded correctly.")
	else:
		error_and_reset(cmd_messages["order_error"], "Order Error", "The architecture is not loaded yet. Make sure to load it before loading the weights.")
		return(-1)



def model_separate_loader():
	model_structure_loader()
	model_weights_loader()



def model_complete_loader():

	global loaded_model, model_complete_file, model_flag, weights_flag

	status_label["text"]=status_model

	model_complete_file = filedialog.askopenfilename(initialdir="/", title="Select Full Model File", filetypes=(("hdf5 files", "*.hdf5"),("all files","*.*")))
	if model_complete_file == '':
		error_and_reset(cmd_messages["file_error"],"File error","No file selected.")
		return(-1)

	if model_flag:
		sys.stdout.write("---\nWARNING: Model already loaded. Attempting to load new model.\n---\n")

	try:
		loaded_model = load_model(model_complete_file, compile=False)
		model_flag = True
		weights_flag = True

		ok_and_reset(cmd_messages["model_load"], model_label, model_text)
		ok_and_reset(cmd_messages["weights_load"], weights_label, weights_text)

	except:
		error_and_reset(cmd_messages["model_error"], "Model Error", "The model couldn't be loaded correctly.")
		return(-1)



def open_img():

	global z, image1, loaded_image, image_file, image_container, canvas1, image_flag

	status_label["text"]=status_image

	image_file = filedialog.askopenfilename(initialdir="/", title="Select Image File", filetypes=(("PNG files", "*.png"),("nrrd files", "*.nrrd"),("all files","*.*")))
	if image_file == '':
		error_and_reset(cmd_messages["file_error"],"File error","No file selected.")
		return(-1)

	if image_file.endswith('.nrrd'):
		try:
			loaded_image, header = nrrd.read(image_file)
		except:
			error_and_reset(cmd_messages["file_error"], "File error","Format not supported.")
			return(-1)

	elif image_file.endswith('.png'):
		try:
			loaded_image = np.array(Image.open(image_file))
		except:
			error_and_reset(cmd_messages["file_error"], "File error","Format not supported.")
			return(-1)

	else:
		error_and_reset(cmd_messages["file_error"], "File error","Format not supported.")
		return(-1)


	if len(loaded_image.shape) > 2:
		z = int(np.floor((loaded_image.shape[2]/7)*4)) # Show arbitrary slice, but at this depth it usually performs well
		image = loaded_image[:,:,z]
	else:
		z = 0
		image = loaded_image[:,:]

	image1 = ImageTk.PhotoImage(Image.fromarray(image))

	if image_flag:
		canvas1.itemconfig(image_container, image=image1)
	else:
		image_container = canvas1.create_image(0,0, image=image1, anchor=NW)
		canvas1.pack_propagate(False)
		canvas1.pack(in_=visualizer, side=LEFT)

	image_flag = True
	ok_and_reset("---\nOPERATION: Loaded image:\n'" + image_file +"'.\n---\n", image_label, image_text)



def segmentation():

	global prediction, image2, loaded_image

	if len(loaded_image.shape) == 2:
		loaded_image.resize(loaded_image.shape[0], loaded_image.shape[1], 1)

	original_shape = loaded_image.shape[:2]
	input_shape = loaded_model.layers[0].input_shape[0][1:3]

	if original_shape != input_shape:

		if (original_shape[0] - input_shape[0] < 0) or (original_shape[1] - input_shape[1] < 0):
			return(-1)

		else:
			sys.stdout.write(f"\n---\nWARNING: The image needs to be resized in order to fit the input shape of the model. A new file with the resized data will be created.\n---\n")
			answer = messagebox.askokcancel(title="Warning: input size does not match", message="The size of the loaded image does not match the input layer size. To solve this, the application will crop the image evenly to match the input size, and a new file with the resized data will be created in the same directory as the original. \nDo you want to continue?")

			if answer != True:
				sys.stdout.write(cmd_messages["stopped"])
				status_reset()
				return(answer)
			
			y = original_shape[0] - input_shape[0]
			x = original_shape[1] - input_shape[1]
    
			loaded_image = loaded_image[int(y/2):-int(y/2),int(x/2):-int(x/2),:]

			if image_file.endswith("png"):
				for i in range(loaded_image.shape[2]):
					cropped_filename = image_file[:-4] + '_cropped'+ str(i).zfill(2) +'.png'
					im = Image.fromarray(loaded_image[:,:,i]).convert('RGB')
					im.save(cropped_filename)
			if image_file.endswith("nrrd"):
				cropped_filename = image_file[:-5] + '_cropped.nrrd'
				nrrd.write(cropped_filename, loaded_image)
			sys.stdout.write("---\nOPERATION: Stored cropped image.\n---\n")


	prediction = np.zeros((loaded_image.shape))

	start = time.time()

	for h in range(loaded_image.shape[2]):

		sys.stdout.write("\r{0}>".format("="*h))
		sys.stdout.flush()

		image = loaded_image[:,:,h]
		image_input = image[np.newaxis, :, :, np.newaxis]
		pred_mask = loaded_model.predict(image_input)[0,:,:,0] > 0.5
		prediction[:,:,h] = pred_mask

		if h == z:
			image2 = ImageTk.PhotoImage(Image.fromarray(pred_mask))

	end = time.time()
	sys.stdout.write(f"\n---\nTIME LOG: Segmentation lasted {end-start}s\n")

	return(0)


def get_prediction():

	global done_flag, canvas2, image2_flag, image2_container

	status_label["text"]=status_prediction

	if ((not model_flag) or (not weights_flag)):
		error_and_reset(cmd_messages["m_w_missing"], "Loading Error", "Model or weights were not loaded.")
		return(-1)

	if not image_flag:
		error_and_reset(cmd_messages["image_missing"], "Loading Error", "No image loaded.")
		return(-1)

	if not check_or_reset():
		return(-1)

	sys.stdout.write("---\nOPERATION: Starting segmentation. Please wait...\n")

	if cpu_flag:
		with tf.device("cpu:0"):
			messagebox.showinfo(title="Device Info", message="The operation will be performed on CPU.")
			res = segmentation()
	else:
		messagebox.showinfo(title="Device Info", message="The operation will be performed on GPU if available.")
		res = segmentation()
	
	if res < 0:
		error_and_reset(cmd_messages["size_error"], "Size error", "Image size does not match the expected input size of the model.")
		return(-1)


	if image2_flag:
		canvas2.itemconfig(image2_container, image=image2)
	else:
		image2_container = canvas2.create_image(0,0, image=image2, anchor=NW)
		canvas2.pack_propagate(False)
		canvas2.pack(in_=visualizer, side=RIGHT)

	image2_flag = True

	done_flag = True
	sys.stdout.write("\n---\nOPERATION: Performed segmentation successfully.\n---\n")
	status_reset()



def save_file():

	global image_file, done_flag

	status_label["text"]=status_save

	if not done_flag:
		error_and_reset(cmd_messages["done_error"],"Segmentation error","No segmentation has been created.")
		return(-1)

	if image_file.endswith(".nrrd"):
		mask_filename = image_file[:-5] + '_predictedMask.nrrd'
		nrrd.write(mask_filename, prediction)

	elif image_file.endswith(".png"):
		mask_filename = image_file[:-4] + '_predictedMask.png'
		print(prediction.shape)
		im = Image.fromarray(prediction[:,:,0]*255).convert('RGB')
		im.save(mask_filename)
	
	sys.stdout.write("---\nOPERATION: Stored segmentation result in directory:\n'" + mask_filename + "'.\n---\n")
	status_reset()





model_flag = False
weights_flag = False
image_flag = False
image2_flag = False

resize_flag = False

done_flag = False

cpu_flag = False

loaded_model = None


# Root
root = Tk()
root.title('Segmentation Model Deployer')
root.geometry("1200x700")

# Menu
menubar = Menu(root)
root.config(menu=menubar)

modelmenu = Menu(menubar, tearoff=0)
modelmenu.add_command(label="Load from one file (.hdf5)", command=model_complete_loader)
modelsubmenu = Menu(menubar, tearoff=0)
modelsubmenu.add_command(label="Load only architecture (.json)", command=model_structure_loader)
modelsubmenu.add_command(label="Load only only weights (.h5)", command=model_weights_loader)
modelsubmenu.add_command(label="Load both", command=model_separate_loader)
modelmenu.add_cascade(label="Load from different files...", menu=modelsubmenu)
modelmenu.add_separator()
modelmenu.add_command(label="Open image (.nrrd)", command=open_img)
menubar.add_cascade(label="File", menu=modelmenu)

menubar.add_command(label="Exit", command=root.quit)

# Objects
cpu_panel = Frame(root)
cpu_panel.pack(side=TOP)
buttons = Frame(root)
buttons.pack(side=TOP)
checklist = Frame(root)
checklist.pack(side=TOP)
status = Frame(root)
status.pack(side=TOP)
visualizer = Frame(root)
visualizer.pack(side=TOP)

canvas1 = Canvas(root, width=576, height=576)
canvas2 = Canvas(root, width=576, height=576)

# Check text
model_text = "Model loaded correctly."
weights_text = "Weights loaded correctly."
image_text = "Image loaded correctly."

model_label = Label(root,
	text = "Model not loaded yet.",
	fg='red')
weights_label = Label(root,
    text = "Weights not loaded yet.",
    fg='red')
image_label = Label(root,
    text = "Image not loaded yet.",
    fg='red')

# Status text
status_waiting = "Status: App waiting..."
status_model = "Status: Loading model..."
status_architecture = "Status: Loading architecture..."
status_weights = "Status: Loading weights..."
status_image = "Status: Loading image..."
status_prediction = "Status: Performing segmentation. Please wait..."
status_save = "Status: Saving image/s..."

status_label = Label(root,
	text = status_waiting)

# Buttons
complete_model_btn = Button(root, text="Load from one file", command=model_complete_loader)
model_btn = Button(root, text="Load model structure", command=model_structure_loader)
weight_btn = Button(root, text="Load model weights", command=model_weights_loader)

open_btn = Button(root, text="Open Image", command=open_img)
pred_btn = Button(root, text="Create Segmentation", command=get_prediction)
save_btn = Button(root, text="Save Segmentation Masks", command=save_file)

cpu_btn = Button(root, text="Force CPU: OFF", command=cpu_switch)

# Packing
model_label.pack(in_=checklist, side=LEFT)
weights_label.pack(in_=checklist, side=LEFT)
image_label.pack(in_=checklist, side=LEFT)

status_label.pack(in_=status, side=LEFT)

complete_model_btn.pack(in_=buttons, side=LEFT)
model_btn.pack(in_=buttons, side=LEFT)
weight_btn.pack(in_=buttons, side=LEFT)
open_btn.pack(in_=buttons, side=LEFT)
pred_btn.pack(in_=buttons, side=LEFT)
save_btn.pack(in_=buttons, side=LEFT)

cpu_btn.pack(in_=cpu_panel, side=RIGHT)

root.mainloop()

