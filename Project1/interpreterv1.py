
from brewparse import parse_program
from intbase import InterpreterBase
from intbase import ErrorType

def main():
	program = """func main() {
				y = "Enger Num";
				x = inputi(y);
				print(x);
			}"""

	interpreter = Interpreter()
	interpreter.run(program)   
  

class Interpreter(InterpreterBase):
	def __init__(self, console_output=True, inp=None, trace_output=False):
		super().__init__(console_output, inp)   # call InterpreterBase's constructor
		self.trace_output = trace_output

	def run(self, program):
		ast = parse_program(program)
		main_func_node = self.get_main_func_node(ast)         # parse program into AST
		self.variable_name_to_value = dict()  # dict to hold variables
		self.run_func(main_func_node)
	

	def get_main_func_node(self, ast):
		functions = ast.dict['functions']
		for function in functions:
			if function.dict['name'] == "main":
				return function
		
		super().error(
    		ErrorType.NAME_ERROR,
    		"No main() function was found",
			)
		return


	def run_func(self, func_node):
		for statement_node in func_node.dict["statements"]:
			self.run_statement(statement_node)

	def run_statement(self, statement_node):
		if statement_node.elem_type == '=': #is_assignment(statement_node)
			self.do_assignment(statement_node)
		elif statement_node.elem_type == 'fcall': #is_func_call(statement_node)
			self.do_func_call(statement_node)



	def do_assignment(self, statement_node):
		target_var_name = statement_node.dict['name']
		source_node = statement_node.dict['expression']

		if self.is_value_node(source_node):
			resulting_value = source_node.dict['val']
		elif self.is_variable_node(source_node):
			if source_node.dict['name'] not in self.variable_name_to_value.keys():
				super().error(
					ErrorType.NAME_ERROR,
					f"Variable " + source_node.dict['name'] + " has not been defined",
				)
				return
			resulting_value = self.variable_name_to_value[source_node.dict['name']]
		else:
			resulting_value = self.evaluate_expression(source_node)

		self.variable_name_to_value[target_var_name] = resulting_value

     

	def evaluate_expression(self, expression_node):
		if self.is_binary_operator(expression_node):
			op1 = expression_node.dict['op1']
			op2 = expression_node.dict['op2']

			op1 = self.reduce_operand(op1)
			op2 = self.reduce_operand(op2)
			
			if (type(op1) != type(op2)) or (type(op1) != int or type(op2) != int):
				super().error(
					ErrorType.TYPE_ERROR,
					"Incompatible types for arithmetic operation",
				)
				return
			
			if expression_node.elem_type == '+':
				return op1 + op2
			else:
				return op1 - op2
		else:
			if expression_node.dict['name'] == 'inputi':
				if len(expression_node.dict['args']) > 1:
					super().error(
    					ErrorType.NAME_ERROR,
    					f"No inputi() function found that takes > 1 parameter",
					)
					return
				
				if len(expression_node.dict['args']) == 1:
					node = expression_node.dict['args'][0]
					if self.is_value_node(node):
						output = node.dict['val']
					elif self.is_variable_node(node):
						if node.dict['name'] not in self.variable_name_to_value.keys():
							super().error(
								ErrorType.NAME_ERROR,
								f"Variable " + node.dict['name'] + " has not been defined",
							)
							return
						output = self.variable_name_to_value[node.dict['name']]
					else:
						output = self.evaluate_expression(node)
					super().output(output)

				user_input = super().get_input()
				return int(user_input)
			else:
				super().error(
					ErrorType.NAME_ERROR,
					f"No function "+ expression_node.dict['name'] + " was found",
				)

		
	def reduce_operand(self, operand):
		if self.is_value_node(operand):
			return operand.dict['val']
		elif self.is_variable_node(operand):
			if operand.dict['name'] not in self.variable_name_to_value.keys():
				super().error(
					ErrorType.NAME_ERROR,
					f"Variable " + operand.dict['name'] + " has not been defined",
				)
				return
			
			return self.variable_name_to_value[operand.dict['name']]
		else:
			return self.evaluate_expression(operand)
		

	def is_value_node(self, expression_node):
		return expression_node.elem_type == 'int' or expression_node.elem_type == 'string'
	
	def is_variable_node(self, expression_node):
		return expression_node.elem_type == 'var'
	
	def is_binary_operator(self, expression_node):
		return expression_node.elem_type == '+' or expression_node.elem_type == '-'
	
	
			
	
	def do_func_call(self, statement_node):
		function_name = statement_node.dict['name']
		if function_name == 'print':
			output_string = ""
			for arg in statement_node.dict['args']:
				if self.is_value_node(arg):
					output_string += str(arg.dict['val'])
				elif self.is_variable_node(arg):
					if arg.dict['name'] not in self.variable_name_to_value.keys():
						super().error(
							ErrorType.NAME_ERROR,
							f"Variable " + arg.dict['name'] + " has not been defined",
						)
						return
					output_string += str(self.variable_name_to_value[arg.dict['name']])
				else:
					output_string += str(self.evaluate_expression(arg))

			super().output(output_string)
		else:
			super().error(
				ErrorType.NAME_ERROR,
				f"No function "+ function_name + " was found",
			)
		


		


if __name__ == "__main__":
    main()

