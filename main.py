import itertools
import kivy
from kivy.properties import ListProperty, OptionProperty
import os
import sys
from kivy.resources import resource_add_path
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from uixwidgets import MatrixValue
from kivy.utils import platform
from kivy.app import App
from kivy.config import Config
from kivy.core.window import Window
import re
from fractions import Fraction

kivy.require('2.0.0')

Config.write()
kivy.resources.resource_add_path("./")


def white_status_bar():
    from android.runnable import run_on_ui_thread 

    @run_on_ui_thread
    def _white_status_bar():
        from jnius import autoclass
        WindowManager = autoclass('android.view.WindowManager$LayoutParams')
        Color = autoclass('android.graphics.Color')
        activity = autoclass('org.kivy.android.PythonActivity').mActivity
        window = activity.getWindow()
        window.clearFlags(WindowManager.FLAG_TRANSLUCENT_STATUS)
        window.addFlags(WindowManager.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
        window.setStatusBarColor(Color.WHITE)
    _white_status_bar()



class MatrixGrid(GridLayout, BoxLayout):
    order = ListProperty([3, 3])


    def on_order(self, *args):

        app.error_list = []

        self.clear_widgets()
        self.rows = int(self.order[0])
        self.cols = int(self.order[1])

        for i in range(1, self.order[0] + 1):
            for k in range(1, self.order[1] + 1):
                text_input = MatrixValue()
                self.add_widget(text_input)

    def show_matrix(self, matrix):
        self.order = [len(matrix), len(matrix[0])]
        unpacked_matrix = list(itertools.chain(*matrix))
        unpacked_matrix.reverse()

        for k in range(0, len(unpacked_matrix)):
            self.children[k].readonly = True
            self.children[k].text = str(unpacked_matrix[k])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_order()



class MatrixCalculator(App):


    operation_config = {'Determinant': ('single', 'square', 'number'),
                        'Rank': ('single', 'any', 'number'),
                        'Addition': ('double', 'same', 'matrix'),
                        'Product': ('double', 'chain', 'matrix'),
                        'Inverse': ('single', 'square', 'matrix')}
    operation_mode = OptionProperty('Determinant', options=operation_config.keys())
    error_list = ListProperty([])
    operation_type = OptionProperty('single', options=['single', 'double'])

    def __init__(self, **kwargs):
        self.title = "Matrix Calculator"
        global app
        app = self
        super().__init__(**kwargs)

    def on_operation_mode(self, *args):
        self.root.ids.display_box.text = ""

    def on_error_list(self, obj, value):
        temp = list(value)
        temp = list(set(self.error_list)) 
        temp = list(filter(None, temp))  
        self.error_list = list(temp)

        length = len(temp)
        if length == 0:
            error_string = ""
        elif length < 4:
            error_string = '\n'.join(self.error_list[:4])
        else:
            error_string = '\n'.join(self.error_list[:3]) + "\n ..."

        self.root.ids.display_box.text = error_string

    def make_matrix(self, matrix):

 
        children_list = matrix.children
        if not children_list:  
            return "---"

        error_observed = False
        self.error_list = []

        for child in children_list: 
            error = Validator().chk_value(child.text)

            if error:
                error_observed = True
                self.error_list.append(error)

        if error_observed:
            return "---"
        else:
            self.error_list = []
            values_list = [Fraction(child.text).limit_denominator(999) for child in children_list]


        values_list.reverse()
        Mvalues_list = []
        temp_list = []
        order = matrix.order

        for i in range(order[0]):
            for k in range(order[1]):
                temp_list.append(values_list[order[1] * i + k])
            Mvalues_list.append(list(temp_list))
            temp_list.clear()

        return Mvalues_list

    
    def calculate(self):

        order_type = self.operation_config[self.operation_mode][1]
        improper_order = Validator().chk_order([self.root.ids.input_matrix_1.order, self.root.ids.input_matrix_2.order], order_type)
        if improper_order:
            return

        matrices_list = [self.make_matrix(self.root.ids.input_matrix_1)]
        if self.operation_config[self.operation_mode][0] == 'double':
            matrices_list.append(self.make_matrix(self.root.ids.input_matrix_2))

        if "---" in matrices_list:
            return

        answer_string = ""
        WHITE_SPACE = "     "

        if self.operation_mode == "Determinant":

            determinant = Calculator().determinant(matrices_list[0])
            answer_string += f"Determinant:{WHITE_SPACE}[anchor='right']{determinant}"

        elif self.operation_mode == "Rank":
            rank = Calculator().rank_of_matrix(matrices_list[0])
            answer_string += f"Rank:{WHITE_SPACE}{rank}"

        elif self.operation_mode == "Addition":
            sum = Calculator().add(matrices_list[0], matrices_list[1])
            answer_string += f"Sum:{WHITE_SPACE}"
            self.root.ids.output_matrix.show_matrix(sum)
            self.root.ids.ans_button.trigger_action()

        elif self.operation_mode == "Product":
            product = Calculator().product(matrices_list[0], matrices_list[1])
            answer_string += f"Product:{WHITE_SPACE}"
            self.root.ids.output_matrix.show_matrix(product)
            self.root.ids.ans_button.trigger_action()

        elif self.operation_mode == "Inverse":
            determinant = Calculator().determinant(matrices_list[0])
            if determinant == 0:
                self.root.ids.display_box.text = ""
                answer_string += "[size=19sp]Inverse not possible for matrix\nwhose determinant is 0.[/size]"
            else:
                inverse = Calculator().inverse(matrices_list[0])
                answer_string += f"Inverse:{WHITE_SPACE}"
                self.root.ids.output_matrix.show_matrix(inverse)
                self.root.ids.ans_button.trigger_action()

        else:
            answer_string += "Choose operation & re-calculate"

      
        self.root.ids.display_box.text = f"[size=29sp]{answer_string}[/size]"

   
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        if platform == "android":
            Window.softinput_mode = 'below_target'  
            white_status_bar()
        else:
            Window.size = (450, 750)  

        return MainWindow()



class MainWindow(BoxLayout):
    pass


class Calculator:

    def sub_matrix(self, A, order):
      
        minors = []
        for i in range(len(A) - order + 1):
            partial_minor = A[i: i + order]
            for k in range(len(A[1]) - order + 1):
                minor = [B[k: k + order] for B in partial_minor]
                minors.append(minor)
        return minors

    def transpose(self, A):
        transposed_matrix = [[k[t] for k in A] for t in range(len(A[0]))]
        return transposed_matrix

    def determinant(self, A, total=0):


        indices = list(range(len(A)))


        if len(A) == 2 and len(A[0]) == 2:
            val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
            return val

        for fc in indices:  
            As = list(A)  
            As = As[1:] 
            height = len(As) 

            for i in range(height):
                
                As[i] = As[i][0:fc] + As[i][fc + 1:]

            sign = (-1) ** (fc % 2)
            sub_det = Calculator().determinant(As)
            total += sign * A[0][fc] * sub_det

        return total

    def inverse(self, A):
        A_copy = list(A)
        det_A = self.determinant(A)
        inversed_matrix = []
        for i in range(len(A)):
            A_copy.pop(i)
            inversed_matrix.append([])
            for j in range(len(A[1])):
                minor_matrix = [k[:j] + k[j + 1:] for k in A_copy]
                cofactor = ((-1) ** (i + j)) * self.determinant(minor_matrix) / det_A
                inversed_matrix[i].append(cofactor)
                print(f"Cofactor = {cofactor} for {minor_matrix}")
            else:
                A_copy = list(A)
        else:
            inversed_matrix = self.transpose(inversed_matrix)

        return inversed_matrix

    def rank_of_matrix(self, A):
        max_rank = min(len(A), len(A[1]))
        print("**** Rank detection started ****")
        for k in reversed(range(1, max_rank + 1)):
            print("on order:", k)
            for f in self.sub_matrix(A, k):
                print("Checking for", f)
                det = self.determinant(f)
                print("Got Determinant:", det)
                if det != 0:
                    print("!! Accepted Rank:", k)
                    return k
        else:
            return 1

    def add(self, A, B):
        summed_matrix = [list(zip(m, n)) for m, n in zip(A, B)]
        for j in range(0, len(summed_matrix)):
            for k in range(0, len(summed_matrix[j])):
                pair = summed_matrix[j][k]
                summed_matrix[j][k] = pair[0] + pair[1]
        print(summed_matrix)
        return summed_matrix

    def product(self, A, B):
        group_by_column = [[k[t] for k in B] for t in range(len(B[0]))]
        print("Grouped by column =", group_by_column)
        product_matrix = []

        for j in A:
            row_matrix = []
            print("=============")

            for k in group_by_column:
                print("=============")
                print(j, '*', k)
                term = [m * n for m, n in zip(j, k)]
                row_matrix.append(sum(term))
                print("Answer =", sum(term))

            else:
                product_matrix.append(row_matrix)

        print("******************************")
        print("Final Product Matrix =", product_matrix)
        return product_matrix


class Validator:
 
    def chk_value(self, value):
    
        value = re.sub(r"\s", "", value) 
        error = None

        master_pattern = re.compile(r"^((\+|\-)?\d{1,3}(([\.]\d{1,2})|([\/]\d{1,3}))?){1}$")

        if not re.match(master_pattern, value):
            if value == '':
                error = "! Any part of matrix can't be EMPTY."
            elif re.search(r"[^\+\-\.\/0-9]", value):
                error = "! Invalid characters in one/more values."
            elif len(re.findall(r"[\/]", value)) > 1:
                error = "! Multiple \'/\' in single value not allowed."
            elif re.search(r"[\/](\+|\-)", value):
                error = "! +/- can be in Numerator, NOT in Denominator."
            elif re.match(r"^((\+|\-)?\d{1,3}([\.]\d)?[\/](\+|\-)?\d{1,3}([\.]\d)?)$", value):
                error = "! Decimal and Fraction can't be together."
            elif re.search(r"\d{4,}", value):
                error = "! Max. 3 digits allowed for numerical part."
            else:
                error = "! Improper structure of entered value/s."

        return error

    def chk_order(self, orders, order_type):
        error = ""

        if order_type == 'square':
            if orders[0][0] != orders[0][1]:
                error = "! Square matrix required for " + app.operation_mode
        elif order_type == 'same':
            if orders[0] != orders[1]:
                error = "! Order of both matrices should be same."
        elif order_type == 'chain':
            if orders[0][1] != orders[1][0]:
                error = "! Columns of M1 should equals to rows of M2"

        if error:
            app.error_list = [error]

        return error

if __name__ == "__main__":
    if hasattr(sys, '_MEIPASS'):
        resource_add_path(os.path.join(sys._MEIPASS))
    MatrixCalculator().run()
