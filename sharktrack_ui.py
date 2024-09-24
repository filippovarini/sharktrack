import customtkinter
import tkinter
from PIL import Image
import os
import re
import send2trash 
import shutil
import configparser

customtkinter.set_appearance_mode("system") 
customtkinter.set_default_color_theme("dark-blue")

# Read in user configuration from 'app_config.ini' file
config = configparser.ConfigParser()
config.read('app_config.ini')
image_path = config['Paths']['Image_Path']
static_path = config['Paths']['Static_Path']
preload_species_name = set(config['Preset Options']['Species_Names'].split('\n'))
preload_species_name.remove('')
print(f"Preloaded species list: {preload_species_name}")

class ModifyFrame(customtkinter.CTkFrame):
    # Name corresponds to the numbers 0, 1 , 2 repectively
    sex_types = ["Female","Male","Unknown"]
    
    def __init__(self, master, **kwargs):
        super().__init__(master, width=420, height=500, **kwargs)
        # This is the App class so we can access its elements
        self.master = master
        self.label = customtkinter.CTkLabel(self, text="Image Details", font=customtkinter.CTkFont(family="Grantha Sangam MN", size=22, weight='bold'))
        self.label.pack(padx=20, pady=15)

        select_box_width = 160
        select_box_height = 35
        self.sex_options = customtkinter.CTkOptionMenu(master=self, values=self.sex_types,
                    corner_radius=10, font=("Roboto", 16), width=select_box_width, height=select_box_height, 
                    fg_color="#1294a7", button_color="#1294a7")
        self.sex_options.pack(padx=20, pady=15)

        # Name corresponds to the index in this list (and passed to data to decode)
        self.species_options = ["Unknown"] 
        if len(preload_species_name) != 0: self.species_options += preload_species_name                           
        self.species_name = customtkinter.CTkComboBox(master=self, values=self.species_options, 
                    font=("Roboto", 16), width=select_box_width, height=select_box_height, corner_radius=10,
                    button_color='#878C8F', border_color='#878C8F')
        self.species_name.pack(padx=20, pady=15)   
 
        self.size_slider_label = customtkinter.CTkLabel(self, text=("Size (ft): \t0.0"), font=("Roboto", 16))
        self.size_slider_label.pack(padx=20, pady=1)  
        
        self.size_slider = customtkinter.CTkSlider(master=self, from_=0, to=20, command=self.slider_event, number_of_steps=40, 
                            button_color="#1294a7", width=200)
        self.size_slider.pack(padx=20, pady=15)  
        self.size_slider_label.configure(text=("Size (ft): \t"+str(self.size_slider.get())))

        current_filename = self.master.loaded_images[self.master.current_file_index]
        if current_filename != "sharktrack_get_started.png":
            self.current_filename_label = customtkinter.CTkLabel(self, text=("Current Filename: \n"+ current_filename), font=("Roboto", 16))
            self.current_filename_label.pack(padx=20, pady=15)

        self.rename_button = customtkinter.CTkButton(master=self, text='Rename (Ctrl N)', command=self.rename_file, 
                    font=("Roboto", 16), width=200, height=32, fg_color="#1294a7")
        self.rename_button.pack(padx=20, pady=20)   

        self.delete_button = customtkinter.CTkButton(master=self, text='Delete (Ctrl ←)', command=self.delete_file, 
                    font=("Roboto", 16), width=200, height=32, fg_color="#878C8F")
        self.delete_button.pack(padx=20, pady=10)   

    def rename_file(self):
        # Declared within App element which has the global current file index
        index_file_to_change = self.master.current_file_index
        current_filename = self.master.loaded_images[index_file_to_change]
        if not os.path.exists(image_path+current_filename): return
        if current_filename == "sharktrack_get_started.png": return

        sex = self.sex_options.get()
        species = self.species_name.get()
        # gets height and removes the decimal point for the file name (re-introudce later on as only one place it can go)
        height_str = str(self.size_slider.get()).replace('.','')

        if not self.sanitise_input(species): return None
        # Add to options list
        if species.lower() not in map(str.casefold, self.species_options):
            # still added to list as list and index can still be passed to MaxN despite not being in UI
            self.species_options.append(species)
            # will stop adding to drop down list so doesn't get too long
            if len(self.species_options) < 12: 
                self.species_name.configure(values=self.species_options)
        
        sex_num = self.sex_types.index(sex)
        species_num = self.species_options.index(species)
        
        # Checks for naming scheme that scripts use
        check_file_reg = re.findall("^\d+-elasmobranch\.jpg$", current_filename)
        if check_file_reg: #can't rename file if named incorrectly
            new_file_name = check_file_reg[0].split('-')[0] + f"-{species_num}-{sex_num}-{height_str}.jpg"
            self.master.next_image(1)
            self.master.loaded_images[index_file_to_change] = new_file_name
            shutil.move((image_path+current_filename), (image_path+new_file_name)) # renames file
            # needed if only one image left
            if len(self.master.loaded_images) == 1:
                self.current_filename_label.configure(text=("Current Filename: \n"+ new_file_name))
            return
        
        # Checks for naming scheme we implement including sex, species, size etc
        check_file_reg = re.findall("^\d+-\d+-\d+-\d+\.jpg$", current_filename)
        if check_file_reg:
            new_file_name = check_file_reg[0].split('-')[0] + f"-{species_num}-{sex_num}-{height_str}.jpg"
            self.master.next_image(1)
            self.master.loaded_images[index_file_to_change] = new_file_name
            shutil.move((image_path+current_filename), (image_path+new_file_name)) # renames file
            # needed if only one image left
            if len(self.master.loaded_images) == 1:
                self.current_filename_label.configure(text=("Current Filename: \n"+ new_file_name))
            return

    def slider_event(self, event):
        self.size_slider_label.configure(text=("Size (ft): \t"+str(self.size_slider.get())))

    def sanitise_input(self, spec_name):
        stripped= ''.join(spec_name.split()) # removes whitespace for sanitisation
        if not stripped.isalpha(): return False
        return True

    def delete_file(self):
        file_path = self.master.current_file_path
        if not os.path.exists(file_path): return

        to_delete = self.master.current_file_index
        if self.master.loaded_images[to_delete] == "sharktrack_get_started.png": return

        # this is if only one file so reset to place holder
        if len(self.master.loaded_images) ==1:
            # delete last image and add placeholder image
            self.master.loaded_images.pop(to_delete)
            self.master.loaded_images.append("sharktrack_get_started.png")

            # update filepath, name and index in main app 
            get_started_filepath = static_path + "sharktrack_get_started.png"
            self.master.current_file_path = get_started_filepath
            self.master.current_file_index= 0

            # display changes
            self.master.main_image.configure(light_image=Image.open(get_started_filepath), size=(768, 492))
            self.master.modify_frame.current_filename_label.configure(text=("Current Filename: \n-"))

            send2trash.send2trash(file_path)
        
        else:
            self.master.next_image(1)
            self.master.loaded_images.pop(to_delete)
            # the current index is incremented in next_image which we would then need to decrement
            # again as popped element, so instead just keep same to_delete index mod len new image
            self.master.current_file_index = to_delete % len(self.master.loaded_images)
            send2trash.send2trash(file_path)
    


class OptionsFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=400, height=500, **kwargs)
        # this is the App class so we can access its elements
        self.master = master

        self.label = customtkinter.CTkLabel(self, text="Options", font=customtkinter.CTkFont(family="Grantha Sangam MN", size=22, weight='bold'))
        self.label.pack(padx=20, pady=15)

        self.maxn_button = customtkinter.CTkButton(master=self, text='Find MaxN', command=self.run_maxn, 
                    font=("Roboto", 16), width=200, height=32, fg_color="#1294a7")
        self.maxn_button.pack(padx=20, pady=10)   

        self.refresh_button = customtkinter.CTkButton(master=self, text='Refresh Images (Ctrl R)', command=master.load_images, 
                    font=("Roboto", 16), width=200, height=32, fg_color="#878C8F")
        self.refresh_button.pack(padx=20, pady=20)   

    
    def run_maxn(self):
        # For development
        None
   


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        #--- Configure tkinter window and images size ---
        screen_width= self.winfo_screenwidth() 
        screen_height= self.winfo_screenheight()
        image_size = (1280, 820)
        image_scale = (screen_width*0.67)/image_size[0]
        self.scaled_image_size = tuple(map(lambda x: x*image_scale, image_size))
        self.geometry("%dx%d" % (screen_width, screen_height))
        self.title("SharkTrack")

        #--- Configure grid system ---
        self.grid_rowconfigure(0, weight=3)  
        self.grid_rowconfigure(1, weight=10)  
        self.grid_rowconfigure(2, weight=6)  
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=6)
        self.grid_columnconfigure(2, weight=2)
        self.grid_columnconfigure(3, weight=6)

        #--- Load Images specified by Image Path  ---
        self.load_images()

        #--- Load Components ---
        self.main_image = customtkinter.CTkImage(light_image=Image.open(self.current_file_path), size=self.scaled_image_size)
        self.image_label = customtkinter.CTkLabel(self, text="", image=self.main_image)
        self.image_label.grid(row=1, column=1, rowspan=2)

        self.modify_frame = ModifyFrame(master=self)
        self.modify_frame.grid(row=1, column=3) 

        self.options_frame = OptionsFrame(master=self)
        self.options_frame.grid(row=2, column=3, sticky="N") 

        self.left_arrow_button = customtkinter.CTkButton(master=self, text='', command=self.left_button_handler, 
                font=("Roboto", 16), width=30, height=32, image=customtkinter.CTkImage(light_image=Image.open((static_path+"left_arrow.png")), size=(22, 30)),
                corner_radius=10, fg_color="#999DA1")
        self.left_arrow_button.grid(row=1, column=0,  rowspan=2, sticky="E") 

        self.right_arrow_button = customtkinter.CTkButton(master=self, text='', command=self.right_button_handler, font=("Roboto", 16), 
                width=25, height=35, image=customtkinter.CTkImage(light_image=Image.open((static_path+"right_arrow.png")), size=(22, 30)),
                corner_radius=10, fg_color="#999DA1")
        self.right_arrow_button.grid(row=1, column=2, rowspan=2, sticky="W") 

        #--- Key binding ---
        self.bind("<Right>", self.right_key_handler)
        self.bind("<Left>", self.left_key_handler)
        self.bind("<Control-BackSpace>", self.back_space_handler)
        self.bind("<Control-n>", self.n_key_handler)
        self.bind("<Control-r>", self.r_key_handler)

    def load_images(self):
        self.loaded_images = []
        for (root, dirs, files) in os.walk(image_path):
            for file in files:
                if '.jpg' in file:
                    self.loaded_images.append(file)
        self.loaded_images.reverse() #to keep in sequential order - acc not rn?? strange
        self.current_file_index = 0

        if len(self.loaded_images) == 0:
            self.loaded_images.append("sharktrack_get_started.png")
            self.current_file_path = os.path.join(static_path, "sharktrack_get_started.png" )
        else:
            try:
                self.loaded_images = sorted(self.loaded_images, key=lambda string: string.split('-')[0])
            except Exception:
                print("Unable to sort images as not following [0-9]- naming scheme")
            self.current_file_path = os.path.join(image_path, self.loaded_images[self.current_file_index])
        
        print(f"Loaded {len(self.loaded_images)} images: {self.loaded_images}")

    # Requires seperate key handling functions to take event parameter
    def right_key_handler(self, event):
        self.next_image(direction=1)

    def left_key_handler(self, event):
        self.next_image(direction=-1)
    
    def back_space_handler(self, event):
        self.modify_frame.delete_file()

    def n_key_handler(self, event):
        self.modify_frame.rename_file()
    
    def r_key_handler(self, event):
        self.load_images()
    
    def right_button_handler(self):
        self.next_image(direction=1)

    def left_button_handler(self):
        self.next_image(direction=-1)

    def next_image(self, direction):
        if self.loaded_images[self.current_file_index] == "sharktrack_get_started.png": return
        # Direction takes +1 for right and -1 for left
        self.current_file_index = (self.current_file_index + direction) % len(self.loaded_images)
        new_filename = self.loaded_images[self.current_file_index]
        self.current_file_path = image_path + new_filename

        self.main_image.configure(light_image=Image.open(self.current_file_path), size=self.scaled_image_size)
        self.modify_frame.current_filename_label.configure(text=("Current Filename: \n"+ new_filename))


app = App()
app.mainloop()
