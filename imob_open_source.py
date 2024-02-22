# coding:utf-8
##******************************************************************************//
## * Project Director: Prof. Lining Zhang                                       //
## * Authors: Ying Ma, Yu Li.                                                   //
## * Notes: This version is utilized for operating iMoB.                        //
## *****************************************************************************//

import tkinter as tk
from tkinter import filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)
from matplotlib.figure import Figure
import os
from PIL import Image, ImageTk
import sys
import pandas as pd
from matplotlib.ticker import MaxNLocator
import re
import subprocess


window = tk.Tk()
window.title('iMoB')
sw = window.winfo_screenwidth()
sh = window.winfo_screenheight()
window.geometry('{}x{}'.format(sw, sh))

# Check if the program is running as a bundled executable
if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS  # This is the temporary directory for the bundled executable
else:
    application_path = os.path.dirname(os.path.abspath(__file__))  # This is the directory of the script

# Build the path to the background.png file
background_path = os.path.join(application_path, './images/background.png')

# Load and scale the image
original_image = Image.open(background_path)
resized_image = original_image.resize((sw, sh), Image.LANCZOS)  # Use LANCZOS for resizing
bg_image = ImageTk.PhotoImage(resized_image)  # Convert the Pillow image to a PhotoImage

# Create a Label to display the image
background_label = tk.Label(window, image=bg_image)
background_label.image = bg_image  # Keep a reference to the image
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# global
test_data_content = None
number_of_columns_output = None
number_of_columns_transfer = None
line_number2  = None
x_output = None
x_tranfer = None

# set two figure
fig = Figure(figsize=(13, 5), dpi=90, layout='constrained')
fig.patch.set_facecolor('#0052A2')
label_style = {'family': 'Arial', 'fontsize': 15, 'weight': 'bold', 'color': 'white'}

# add first figure
ax1 = fig.add_subplot(121)
ax1.set_xlabel("X-axis", **label_style)
ax1.set_ylabel("Y-axis", **label_style)

# add two figure
ax2 = fig.add_subplot(122)
ax2.set_xlabel("X-axis", **label_style)
ax2.set_ylabel("Y-axis", **label_style)



# set tick color
ax1.tick_params(axis='both', direction='in', labelsize=15, labelcolor='white')
ax2.tick_params(axis='both', direction='in', labelsize=15, labelcolor='white')


# set tick in
ax1.tick_params(direction='in', length=5, width=2, colors='white', grid_color='#0052A2', grid_alpha=1)
ax2.tick_params(direction='in', length=5, width=2, colors='white', grid_color='black', grid_alpha=0.5)


# Set the number of tick
ax1.xaxis.set_major_locator(MaxNLocator(10))
ax1.yaxis.set_major_locator(MaxNLocator(10))
ax2.xaxis.set_major_locator(MaxNLocator(10))
ax2.yaxis.set_major_locator(MaxNLocator(10))

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().place(x=150, y=250)

# data loading function
def load_():
    global test_data_content, number_of_columns_output,number_of_columns_transfer,line_number2,x_output,x_tranfer

    Path1 = tk.filedialog.askopenfilename()
    print('\nfile path:', Path1)
    file = Path1
    handle = open(file, mode='r')
    # read file line by line
    def find_first_two_functions(file_path):
        global line_number1, line_number2
        line_number1 = None
        line_number2 = None
        with open(file_path, 'r') as handle:
            for line_number, line in enumerate(handle, start=1):
                columns = line.split('\t')
                if 'function' in columns:
                    if line_number1 is None:
                        line_number1 = line_number  # save the line number of the first 'function
                    elif line_number2 is None:
                        line_number2 = line_number  # save the line number of the second 'function
                        break  # stop
        return line_number1, line_number2

    line_number1, line_number2 = find_first_two_functions(Path1)

    #Import file (skip lines with characters)
    row_to_skip=list(range(line_number1-1, line_number1+2)) + list(range(line_number2-1,line_number2+2))
    test_data_content=pd.read_csv(file, delimiter='\t', skiprows=row_to_skip)
    #delect NAN
    test_data_content = test_data_content.dropna(axis=1, how='all')
    #convert to numeric type
    test_data_content = test_data_content.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
   #for idvd.csv
    def count_columns_in_file(file_path, delimiter='\t', line_number=line_number1+3):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == line_number - 1:
                    # number of  '('
                    return line.count('(')

    def extract_values_from_line1(line):
        # find vg=**
        pattern = re.compile(r'Id\(vg=(0|\d+\.\d+)\)')
        values = pattern.findall(line)
        return values

    def read_line_from_file(file_path, line_number):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == line_number:
                    return line  #
    # read line of vg=**
    line_content_o = read_line_from_file(Path1, line_number1 + 2)
    values_in_brackets1 = extract_values_from_line1(line_content_o)
    # Create global variables vd1, vd2, ..., vdn and assign values
    for index, value in enumerate(values_in_brackets1, start=1):
        globals()[f'vg{index}'] = value

    #  Call function
    number_of_columns_output = count_columns_in_file(Path1)
    #print (number_of_columns_output)
    #find the total number of columns
    def count_columns_in_file_b(file_path, line_number=line_number2 + 3):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == line_number - 1:
                    return line.count('(')

    def extract_values_from_brackets(line):
        pattern = re.compile(r'Id\(vd=(\d+\.\d+)\)')
        values = pattern.findall(line)
        return values

    def read_line_from_file(file_path, line_number):
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i == line_number:
                    return line

    number_of_columns_transfer = count_columns_in_file_b(Path1)
    # read and extract the values of specific rows
    line_content = read_line_from_file(Path1, line_number2 + 2)
    values_in_brackets = extract_values_from_brackets(line_content)
    #  Create global variables vd1, vd2, ..., vdn and assign values
    for index, value in enumerate(values_in_brackets, start=1):
        globals()[f'vd{index}'] = value

# Plot function
    if test_data_content is not None:
        # use the first column as the x-axis
        x_output = test_data_content.iloc[0:line_number2 - 5, 0]
        #print(x_output)

        # initialize y_min and y_max
        y_min, y_max = float('inf'), float('-inf')
        #set the first column to display labels
        y1 = test_data_content.iloc[0:line_number2 - 5, 1]
        ax1.plot(x_output, y1, marker='o', linestyle='', color='blue', markerfacecolor='none', markeredgewidth=1,
                 label=f'Data')

        # use the remaining columns as the y-axis without displaying labels
        for i in range(2, number_of_columns_output):
            y = test_data_content.iloc[0:line_number2 - 5, i]

            # update y_min and y_max
            y_min = min(y_min, y.min())
            y_max = max(y_max, y.max())
            ax1.plot(x_output, y, marker='o', linestyle='', color='blue', markerfacecolor='none', markeredgewidth=1)

        # expand the chart range
        y_range = y_max - y_min
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # set the Y-axis range of ax1
        ax1.set_ylim(y_min, y_max)

        # set title, axis labels, and legend
        ax1.set_title('Output', color='white')
        ax1.set_xlabel('Vds')
        ax1.set_ylabel('Ids')
        ax1.legend()
        #plotting idvd.csv
        # set vg data as the first column
        repeated_column = pd.concat([x_output] * number_of_columns_output, ignore_index=True)
        # set vd data as the second column
        repeated_values = [value for value in values_in_brackets1 for _ in range(line_number2 - 5)]
        # initialize an empty list to save id
        concatenated_column = []

        # iterate through columns and add data to the list one by one
        for i in range(1, number_of_columns_output + 1):
            y = test_data_content.iloc[0:line_number2 - 5, i]
            # format into scientific notation and add to the list
            formatted_values = [f"{value:.2e}" for value in y.values]
            concatenated_column.extend(formatted_values)

        #print("Size of concatenated_column:", len(concatenated_column))
        # create a new IDVD DataFrame
        new_df = pd.DataFrame({
            'vg': repeated_column,  #
            'vd': repeated_values, #
            'id': concatenated_column  #
        })

        # save as idvd.csv
        new_df.to_csv('idvd.csv', index=False, header=False)

        # for the second plot, use the first column as the x-axis
        x_tranfer = test_data_content.iloc[line_number2 - 4:2 * line_number2 - 9, 0]

        #set the first column to have labels
        y2 = test_data_content.iloc[line_number2 - 4:2 * line_number2 - 9, 1]
        # draw the graph with blue lines and hollow circle markers
        ax2.plot(x_tranfer, y2, marker='o', linestyle='', color='blue', markerfacecolor='none', markeredgewidth=1,
                 label=f'Data')
        for i in range(2, number_of_columns_transfer):
            y3 = test_data_content.iloc[line_number2 - 4:2 * line_number2 - 9, i]
            ax2.plot(x_tranfer, y3, marker='o', linestyle='', color='blue', markerfacecolor='none', markeredgewidth=1,
                     )
        # set title, axis labels, and legend
        ax2.set_yscale('log')
        ax2.set_title('Transfer', color='white')
        ax2.set_xlabel('Vgs')
        ax2.set_ylabel('Ids')
        ax2.legend()
        # idvg.csv data
        # set vg data as the first column
        repeated_column = pd.concat([x_tranfer] * number_of_columns_transfer, ignore_index=True)

        # set vd data as the second column
        repeated_values = [value for value in values_in_brackets for _ in range(line_number2 - 5)]

        # set id data
        # initialize an empty list to save id
        concatenated_column = []

        # iterate through columns and add data to the list one by one
        for i in range(1, number_of_columns_transfer + 1):
            y = test_data_content.iloc[line_number2 - 4:2 * line_number2 - 9, i]
            concatenated_column.extend(y)  # add column data to the list

            # idvg DataFrame
        new_df = pd.DataFrame({
            'vg': repeated_values,
            'vd': repeated_column,
            'id': concatenated_column
        })

        # save idvg.csv
        new_df.to_csv('idvg.csv', index=False, header=False)
    canvas.draw()

b_load = tk.Button(window, text='Load Data', width=15, height=2, command=load_)
b_load.place(x=150, y=200)

script_path = tk.StringVar()
def load_():
    global script_path, file_container, file_content,content_y_nn, number_of_columns_output,number_of_columns_transfer,line_number2,x_output,x_tranfer

    # load data.py
    Path2 = filedialog.askopenfilename()
    print('\nfile path:', Path2)

    file_container = Path2[:Path2.rindex('/')]
    #print('file container: ', file_container)
    print("In the process of training...")


    # run script.py
    file = Path2
    script_path.set(Path2)

    with open(file, mode='r', encoding='utf-8') as f:
        file_content = f.read()
    subprocess.run(['python', Path2], check=True)

    file_y = file_container + '/current_predicted.txt'
    content_y_nn = np.loadtxt(file_y)
    if content_y_nn is not None:
        # first figure
        # plot first figure
        # set the first column to have labels
        ax1.plot(x_output, content_y_nn[(line_number2 - 5):(2) * (line_number2 - 5)], c='r', label=f'Predicted')
        # set the remaining columns without labels and plot
        for i in range(number_of_columns_output - 1):
            ax1.plot(x_output, content_y_nn[i * (line_number2 - 5):(i + 1) * (line_number2 - 5)], c='r')
        ax1.set_title('Output', color='white')
        ax1.set_xlabel('Vds')
        ax1.set_ylabel('Ids')
        ax1.legend()

        # second figure
        # plot second figure
        # set the first column to have labels
        ax2.plot(x_tranfer, content_y_nn[(1 + number_of_columns_output) * (line_number2 - 5):(2 + number_of_columns_output) * (
                    line_number2 - 5)], c='r', label=f'Predicted')
        # set the remaining columns without labels and plot
        for i in range(number_of_columns_transfer - 1):
            ax2.plot(x_tranfer, content_y_nn[(i + number_of_columns_output) * (line_number2 - 5):(i + 1 + number_of_columns_output) * (line_number2 - 5)], c='r')
        ax2.set_yscale('log')
        ax2.set_title('Transfer', color='white')
        ax2.set_xlabel('Vgs')
        ax2.set_ylabel('Ids')
        ax2.legend()
    canvas.draw()
b_load_script = tk.Button(window, text='Train Script', width=15, height=2, command=load_)
b_load_script.place(x=300, y=200)
window.mainloop()
