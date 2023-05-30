import re
import os
def insert_line_breaks(text):
    # Use regular expression to find periods followed by a space and insert line breaks
    modified_text = re.sub(r'\. ', '.\n', text)

    # save to file
    with open("b.txt", "w") as f:
        f.write(modified_text)
        
    return modified_text

# Example usage
with open("a.txt", "r") as f:
    input_text = f.read()
output_text = insert_line_breaks(input_text)
print(output_text)