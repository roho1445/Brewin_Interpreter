import copy
from enum import Enum

from brewparse import parse_program
from env_v2 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev2 import Type, Value, create_value, get_printable
from Lambda_Node import Lambda


class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

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
        self.env = EnvironmentManager()
        #self.lambda_envs = {}
        main_func = self.__get_func_by_name("main", 0)
        self.__run_statements(main_func.get("statements"))

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params = -1):
        if name not in self.func_name_to_ast.keys():
            var_func = self.env.get(name) 
            if var_func == None:
                super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
            try:
                if var_func.elem_type == "function" or var_func.elem_type == "lambda":
                    pass
            except:
                super().error(ErrorType.TYPE_ERROR, f"Function {name} is a not a function")

            if len(var_func.get("args")) != num_params:
                super().error(ErrorType.TYPE_ERROR, f"Function {name} has incorrect parameters inputted")
            return var_func
        
        candidate_funcs = self.func_name_to_ast[name]
        if num_params == -1:
            if len(candidate_funcs) != 1:
                super().error(
                ErrorType.NAME_ERROR,
                f"Overloaded function assigned to variable",
            )
            return list(candidate_funcs.values())[0]
        
        if num_params not in candidate_funcs:
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {name} taking {num_params} params not found",
            )
        return candidate_funcs[num_params]

    def __run_statements(self, statements):
        self.env.push()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status = ExecStatus.CONTINUE
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == InterpreterBase.RETURN_DEF:
                status, return_val = self.__do_return(statement)
            elif statement.elem_type == Interpreter.IF_DEF:
                status, return_val = self.__do_if(statement)
            elif statement.elem_type == Interpreter.WHILE_DEF:
                status, return_val = self.__do_while(statement)

            if status == ExecStatus.RETURN:
                self.env.pop()
                return (status, return_val)

        self.env.pop()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_input(call_node)
        if func_name == "inputs":
            return self.__call_input(call_node)
        #temp_env = copy.deepcopy(self.env)
        actual_args = call_node.get("args")
        func_ast = self.__get_func_by_name(func_name, len(actual_args))
        formal_args = func_ast.get("args")
        
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.NAME_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )

        self.env.push()

        #Load Paramters
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            if formal_ast.elem_type == InterpreterBase.ARG_DEF:
                result = copy.deepcopy(self.__eval_expr(actual_ast))
            else:
                result = self.__eval_expr(actual_ast)
            arg_name = formal_ast.get("name")
            self.env.create(arg_name, result)
        
        #Load Captured Lambda Vals if applicable
        islambda = False
        if func_ast.elem_type == InterpreterBase.LAMBDA_DEF:
            islambda = True
            self.env.push()
            lambdaEnvDict = func_ast.lambda_env.getEnvDict()
            #currEnvDict = self.env.getEnvDict()
            formal_arg_names = []
            for formal_ast in formal_args:
                formal_arg_names.append(formal_ast.get("name"))
            for key in lambdaEnvDict.keys():
                if key not in formal_arg_names:
                    self.env.create(key, func_ast.lambda_env.get(key))
            
        
        status, return_val = self.__run_statements(func_ast.get("statements"))

        if islambda:
            popped_env = self.env.pop()
            for key, value in popped_env.items():
                func_ast.lambda_env.set(key, value)

        popped_env = self.env.pop()
        #self.env = temp_env
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            if formal_ast.elem_type == InterpreterBase.REFARG_DEF:
                if actual_ast.elem_type == InterpreterBase.VAR_DEF:
                    self.env.set(actual_ast.get("name"), copy.deepcopy(popped_env[formal_ast.get("name")]))

        #print("function env: " + func_name)
        #print(return_val.lambda_env.getEnvDict())
        return return_val

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
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
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
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            val = self.env.get(var_name)
            if val is None:
                return self.__get_func_by_name(var_name)
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == Interpreter.NEG_DEF:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        if expr_ast.elem_type == Interpreter.NOT_DEF:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)
        if expr_ast.elem_type == InterpreterBase.LAMBDA_DEF:
            return Lambda(expr_ast, copy.deepcopy(self.env))


    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))

        #Function Comparison
        if arith_ast.elem_type == '==':
            try:
                if left_value_obj.elem_type in [InterpreterBase.LAMBDA_DEF, InterpreterBase.FUNC_DEF]:
                    return Value(Type.BOOL, left_value_obj is right_value_obj)
            except:
                pass
            try:
                if right_value_obj.elem_type in [InterpreterBase.LAMBDA_DEF, InterpreterBase.FUNC_DEF]:
                    return Value(Type.BOOL, left_value_obj is right_value_obj)
            except:
                pass
        elif arith_ast.elem_type == '!=':
            try:
                if left_value_obj.elem_type in [InterpreterBase.LAMBDA_DEF, InterpreterBase.FUNC_DEF]:
                    return Value(Type.BOOL, left_value_obj is not right_value_obj)
            except:
                pass
            try:
                if right_value_obj.elem_type in [InterpreterBase.LAMBDA_DEF, InterpreterBase.FUNC_DEF]:
                    return Value(Type.BOOL, left_value_obj is not right_value_obj)
            except:
                pass

        try:
            typetemp = left_value_obj.type()
        except:
            super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {arith_ast.elem_type} operation",
                )
        
        try:
            typetemp = right_value_obj.type()
        except:
            super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible types for {arith_ast.elem_type} operation",
                )
        

        if arith_ast.elem_type in ['==','!=']:
            if left_value_obj.type() == Type.BOOL or right_value_obj.type() == Type.BOOL:
                left_value_obj = self.convert_int_to_bool(left_value_obj)
                right_value_obj = self.convert_int_to_bool(right_value_obj)
        
        if arith_ast.elem_type in ['&&', '||']:
            left_value_obj = self.convert_int_to_bool(left_value_obj)
            right_value_obj = self.convert_int_to_bool(right_value_obj)
        
        if arith_ast.elem_type in ['+', '-', '*', '/']:
            left_value_obj = self.convert_bool_to_int(left_value_obj)
            right_value_obj = self.convert_bool_to_int(right_value_obj)
        
        
        
        if not self.__compatible_types(arith_ast.elem_type, left_value_obj, right_value_obj):
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
    
    def convert_int_to_bool(self, val_ast):
        try:
            if val_ast.type() != Type.INT:
                return val_ast
        except:
            super().error(
                ErrorType.TYPE_ERROR,
                "Functions cannot be converted",
            )
        if val_ast.value() == 0:
            return Value(Type.BOOL, False)
        else:
            return Value(Type.BOOL, True)
    
    def convert_bool_to_int(self, val_ast):
        try:
            if val_ast.type() != Type.BOOL:
                return val_ast
        except:
            super().error(
                ErrorType.TYPE_ERROR,
                "Functions cannot be converted",
            )
        if val_ast.value() == True:
            return Value(Type.INT, 1)
        else:
            return Value(Type.INT, 0)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if arith_ast.elem_type == InterpreterBase.NOT_DEF:
            value_obj = self.convert_int_to_bool(value_obj)
        if value_obj.type() != t:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible type for {arith_ast.elem_type} operation",
            )
        return Value(t, f(value_obj.value()))

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
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )
        #  set up operations on strings
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
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            x.type(), x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            x.type(), x.value() or y.value()
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.convert_int_to_bool(self.__eval_expr(cond_ast))
        try:
            typetemp = result.type()
        except:
            super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for if statement",
                )
        if result.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if result.value():
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_while(self, while_ast):
        cond_ast = while_ast.get("condition")
        run_while = Interpreter.TRUE_VALUE
        while run_while.value():
            run_while = self.convert_int_to_bool(self.__eval_expr(cond_ast))
            try:
                typetemp = run_while.type()
            except:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for while loop",
                )
            if run_while.type() != Type.BOOL:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for while condition",
                )
            if run_while.value():
                statements = while_ast.get("statements")
                status, return_val = self.__run_statements(statements)
                if status == ExecStatus.RETURN:
                    return status, return_val

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        value_obj = copy.deepcopy(self.__eval_expr(expr_ast))
        return (ExecStatus.RETURN, value_obj)
    

if __name__ == "__main__":
    program = """
func main() {
  b = lambda(a){ return lambda(b){ return lambda(c){return a + b + c;}; }; };
  if(b)
  {
    print("hi");
  }
  
}

			    """
    
    interpreter = Interpreter()
    interpreter.run(program)
