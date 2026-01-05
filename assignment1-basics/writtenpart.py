import sys



def problem_1a():
    print(chr(0))


def problem_1b():
    print(chr(0).__repr__())


def problem_1c():
    print("Testing chr(0):")
    print("chr(0) =", chr(0))
    print("print(chr(0)) =", end=" ")
    print(chr(0))
    print()
    
    test_string = "this is a test" + chr(0) + "string"
    print("String with chr(0):", repr(test_string))
    print("print() of that string:", end=" ")
    print(test_string)


def problem_2b():

    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])

    try:
        word = "café"
        print(decode_utf8_bytes_to_str_wrong(word.encode("utf-8")))
    except Exception as e:
        print("Found a Word that cannot be decoded: ", word)

def problem_2c():

    two_bytes=b"\x80\x80"

    try:
        print(two_bytes.decode("utf-8"))
    except Exception as e:
        print("Found a two byte sequence that cannot be decoded: ", two_bytes)

if __name__ == "__main__":

    if sys.argv[1] == "1a":
        problem_1a()

    elif sys.argv[1] == "1b":
        problem_1b()

    elif sys.argv[1] == "1c":
        problem_1c()

    elif sys.argv[1] == "2b":
        problem_2b()

    elif sys.argv[1] == "2c":
        problem_2c()


