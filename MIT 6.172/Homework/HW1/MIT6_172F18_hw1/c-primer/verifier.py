#!/usr/bin/python
import subprocess, os, sys, re

def exitWithError(error):
    print error
    sys.exit(1)

def runAndReadOutput(args):
    if args is str:
        args = [args]
    try:
        return subprocess.check_output(args)
    except subprocess.CalledProcessError as e:
        exitWithError("ERROR: runtime error with %s" % str(args))

def run(path): runAndReadOutput(path)

def runAndCheckSizes():
    output = runAndReadOutput("./sizes")
    expected_output_format = "size of %s : %d bytes"

    lines = set([x.replace(" ", "") for x in output.strip().lower().split('\n')])
    types = [
        ( "int", 4 ),
        ( "short", 2 ),
        ( "long", 8 ),
        ( "char", 1 ),
        ( "float", 4 ),
        ( "double", 8 ),
        ( "unsigned int", 4 ),
        ( "long long", 8 ),
        ( "uint8_t", 1 ),
        ( "uint16_t", 2 ),
        ( "uint32_t", 4 ),
        ( "uint64_t", 8 ),
        ( "uint_fast8_t", 1 ),
        ( "uint_fast16_t", 8 ),
        ( "uintmax_t", 8 ),
        ( "intmax_t", 8 ),
        ( "__int128", 16 ),
        ( "uint32_t", 4 ),
        ( "uint64_t", 8 ),
        ( "student", 8 ),
	( "x", 20),
        ( "int*", 8 ),
        ( "short*", 8 ),
        ( "long*", 8 ),
        ( "char*", 8 ),
        ( "float*", 8 ),
        ( "double*", 8 ),
        ( "unsigned int*", 8 ),
        ( "long long*", 8 ),
        ( "uint8_t*", 8 ),
        ( "uint16_t*", 8 ),
        ( "uint32_t*", 8 ),
        ( "uint64_t*", 8 ),
        ( "uint_fast8_t*", 8 ),
        ( "uint_fast16_t*", 8 ),
        ( "uintmax_t*", 8 ),
        ( "intmax_t*", 8 ),
        ( "__int128*", 8 ),
        ( "uint32_t*", 8 ),
        ( "uint64_t*", 8 ),
        ( "student*", 8 ),
	( "&x", 8)
    ]

    for typ in types:
        print (expected_output_format % typ)
        if (expected_output_format % typ).replace(" ", "") not in lines:
            exitWithError("ERROR: couldn't find type %s (or it has the incorrect value) in sizes output" % typ[0])

def runAndCheckSwap():
    expected_output = "k = 2, m = 1\n"
    output = runAndReadOutput("./swap")

    if output != expected_output:
        exitWithError('ERROR: actual output: "%s", expected "%s"' % (output, expected_output))

def build(make_arg, filename):
    print "\nRunning make %s ... " % make_arg
    run(["make", filename])
    print "Ok!"

    print "\nChecking that %s was built ... " % filename
    if not os.path.isfile(filename):
        exitWithError("ERROR: %s binary missing, did you rename it?" % filename)
    print "Ok!"


print "Running verifying script ... "

print "\nChecking that the Makefile exists ... "
if not os.path.isfile('Makefile'):
    exitWithError('ERROR: Makefile does not exist.')
print "Good!"

build("sizes", "sizes")
print "Checking output of sizes ... "
runAndCheckSizes()
print "Ok!"

build("pointer", "pointer")
run("./pointer")  # Run pointer as a sanity check, but there's no output to check

build("swap", "swap")
print "Checking output of swap ... "
runAndCheckSwap()
print "Ok!"

print "LGTM"
