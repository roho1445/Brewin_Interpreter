from env_v1 import EnvironmentManager
from type_valuev1 import Type, Value, create_value, get_printable
from intbase import InterpreterBase, ErrorType
from brewparse import parse_program
import copy


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    BIN_OPS = {"+", "-", "*", "/"}
    COMP_OPS = {"==", "!=", ">", ">=", "<", "<="}
    BOOL_OPS = {"&&", "||"}
    UNARY_OPS = {InterpreterBase.NOT_DEF, InterpreterBase.NEG_DEF}


    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()


    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        main_func = self.__get_func_by_name("main")
        self.env = EnvironmentManager()
        self.__run_statements(main_func.get("statements"))


    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            numArgs = len(func_def.get("args"))
            func_name = func_def.get("name")
            if func_def.get("name") != "main":
                func_name = func_def.get("name") + str(numArgs)
            self.func_name_to_ast[func_name] = func_def


    def __get_func_by_name(self, name):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        return self.func_name_to_ast[name]


    def __run_statements(self, statements):
        x = ""
        for statement in statements:
            if self.trace_output:
                print(statement)
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == Interpreter.IF_DEF:
                ifRet = self.__eval_if(statement)
                if ifRet.returnedMidway():
                    return ifRet
            elif statement.elem_type == Interpreter.WHILE_DEF:
                whileRet = self.__eval_while(statement)
                if whileRet.returnedMidway():
                    return whileRet
            elif statement.elem_type == Interpreter.RETURN_DEF:
                x = copy.deepcopy(self.__eval_return(statement))
                x.r = True
                return x
        return copy.deepcopy(Interpreter.NIL_VALUE)
    

    def __eval_return(self, return_node):
        if return_node.get("expression") == None:
            return Interpreter.NIL_VALUE
        
        return self.__eval_expr(return_node.get("expression"))


    def __eval_while(self, while_node):
        envCopy = copy.deepcopy(self.env)
        x = Interpreter.NIL_VALUE
        whileClause = self.__eval_expr(while_node.get("condition"))
        if whileClause.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR, "Incorrect type for while statement"
            )
        while whileClause.value() == True:
            x = self.__run_statements(while_node.get("statements"))
            if x.returnedMidway():
                break
            whileClause = self.__eval_expr(while_node.get("condition"))
            if whileClause.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR, "Incorrect type for while statement"
                )
        for key in envCopy.environment.keys():
            envCopy.set(key, self.env.get(key))
        self.env = envCopy
        return x


    def __eval_if(self, if_node):
        envCopy = copy.deepcopy(self.env)
        x = Interpreter.NIL_VALUE
        ifClause = self.__eval_expr(if_node.get("condition"))
        if ifClause.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR, "Incorrect type for if statement"
            )

        if ifClause.value() == True:
            x = self.__run_statements(if_node.get("statements"))
        elif if_node.get("else_statements") != None:
            x = self.__run_statements(if_node.get("else_statements"))

        for key in envCopy.environment.keys():
            envCopy.set(key, self.env.get(key))
        
        self.env = envCopy
        return x


    def __call_func(self, call_node):
        func_name = call_node.get("name")
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi" or func_name == "inputs":
            return self.__call_input(call_node)
        argInputs = call_node.get("args")
        func_name = func_name + str(len(argInputs))
        func_node = self.__get_func_by_name(func_name)
        argNodes = func_node.get("args")
        envCopy = copy.deepcopy(self.env)
        original_arg_vals = dict()
        for i in range(len(argInputs)):
            if self.env.get(argNodes[i].get("name")) != None:
                original_arg_vals[argNodes[i].get("name")] = self.env.get(argNodes[i].get("name"))
            self.env.set(argNodes[i].get("name"), copy.deepcopy(self.__eval_expr(argInputs[i])))
        retVal = self.__run_statements(func_node.get("statements"))
        
        for key in envCopy.environment.keys():
            envCopy.set(key, self.env.get(key))


        for key, value in original_arg_vals.items():
            envCopy.set(key, value)
        
        self.env = envCopy
        return retVal
        
    
    def __call_print(self, call_ast):
        output = ""
        for arg in call_ast.get("args"):
            result = self.__eval_expr(arg)  # result is a Value object
            output = output + get_printable(result)
        super().output(output)
        return Interpreter.NIL_VALUE


    def __call_input(self, call_ast):
        args = call_ast.get("args")
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No input function that takes > 1 parameter"
            )
        inp = super().get_input()
        if call_ast.get("name") == "inputi":
            return Value(Type.INT, int(inp))
        if call_ast.get("name") == "inputs":
            return Value(Type.STRING, inp)


    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))
        self.env.set(var_name, value_obj)


    def __eval_expr(self, expr_ast):
        if expr_ast == None:
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            return Value(Type.NIL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            val = self.env.get(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            x = self.__call_func(expr_ast)
            return x
        if expr_ast.elem_type in Interpreter.BIN_OPS or expr_ast.elem_type in Interpreter.COMP_OPS or expr_ast.elem_type in Interpreter.BOOL_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type in Interpreter.UNARY_OPS:
            return self.__eval_unary(expr_ast)
    

    def __eval_unary(self, unary_ast):
        value_obj = self.__eval_expr(unary_ast.get("op1"))
        f = self.op_to_lambda[value_obj.type()][unary_ast.elem_type]
        return f(value_obj)


    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        if left_value_obj.type() != right_value_obj.type() and arith_ast.elem_type != "==" and arith_ast.elem_type != "!=":
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        return f(left_value_obj, right_value_obj)
    

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["neg"] = lambda x: Value(
            x.type(), x.value() * -1
        )
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        # set up operations on booleans
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            x.type(), x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            x.type(), x.value() != y.value()
        )
        self.op_to_lambda[Type.BOOL]["!"] = lambda x: Value(
            x.type(), not x.value()
        )
        # set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        # set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )


if __name__ == "__main__":
    program = """func main() {
  value1 = 11;
  value2 = 20;
  value3 = " entered";
  print(value2);


  if (value1 < 20){
    x = 1;
    x = "hi";
    print( == "hi");
  }
  print(value1);
}
			    """
    
    interpreter = Interpreter()
    interpreter.run(program)