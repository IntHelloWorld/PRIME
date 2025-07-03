class FunctionArgumentError(Exception):
    def __init__(self, function_name):
        self.message = f"Function {function_name} cannot be called with the given arguments. Please check the function arguments you provided and try again."

    def __str__(self):
        return {self.message}


class ClassNotFoundError(Exception):
    def __init__(self, class_name):
        self.message = f"No covered methods in the class {class_name} were found. The class may not covered by the failing tests. Please check your input or try something else."

    def __str__(self):
        return {self.message}


class ClassNameNotFoundError(Exception):
    def __init__(self, short_class_name):
        self.message = f"No classes with the name {short_class_name} were found. Please try something else."

    def __str__(self):
        return {self.message}


class MethodNotFoundError(Exception):
    def __init__(self, method_name):
        self.message = f"No methods with the name {method_name} were found. Please check your try something else."

    def __str__(self):
        return {self.message}


class MethodIDNotFoundError(Exception):
    def __init__(self, method_id):
        self.message = f"No covered methods with the ID '{method_id}' were found. Please check your input or try something else."

    def __str__(self):
        return {self.message}


class ArgumentError(Exception):
    def __init__(self):
        self.message = f"Wrong augument format for the tool! Please follow the format and try again."

    def __str__(self):
        return {self.message}


class SnippetNotFoundError(Exception):
    def __init__(self, code_snippet):
        self.message = f"No covered methods contain the code snippet were found. Please check your input or try something else."
        self.code_snippet = code_snippet

    def __str__(self):
        return f"{self.message}\n\n{self.code_snippet}"


class EditCommandFormatError(Exception):
    def __init__(self, edit_command):
        self.message = "Wrong *SEARCH/REPLACE* edit format! please follow the format and try again."
        self.edit_command = edit_command

    def __str__(self):
        return f"{self.message}\n\n{self.edit_command}"


class EditCommandContentError(Exception):
    def __init__(self, edit_command):
        self.message = (
            "Wrong *SEARCH/REPLACE* edit content! We cannot find the code you want to replace, please mind that:\n"
            "- You should **re-edit the suspicious Java method from scratch** each time\n"
            "- Check the indentations or line breaks"
        )
        self.edit_command = edit_command

    def __str__(self):
        return f"{self.message}\n\n{self.edit_command}"


class PrintStmtNotFoundError(Exception):
    def __init__(self, content):
        self.message = "Cannot find the `System.out.println` statements in the edited method. Please try again."
        self.content = content

    def __str__(self):
        return f"{self.message}\n\n{self.content}"


class CompileError(Exception):
    def __init__(self, compile_message):
        self.message = f"There are errors to compile and run the edited method, you shouldn't modified the original code functionality or use non-existent variables. Please check the following error message and try again:\n{compile_message}"

    def __str__(self):
        return self.message


class PrintDebugFileNotFoundError(Exception):
    def __init__(self):
        self.message = "Cannot find the suspicious method to perform the print debugging. Note that you should not perform the print debugging on a Java test method!"

    def __str__(self):
        return self.message
