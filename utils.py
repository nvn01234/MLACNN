import traceback


def make_dict(*expr):
    (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
    begin = text.find('make_dict(') + len('make_dict(')
    end = text.find(')', begin)
    text = [name.strip() for name in text[begin:end].split(',')]
    return dict(zip(text, expr))