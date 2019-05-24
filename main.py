import tasks
import MNIST

def main():

    #Task 1 runs the first task about EM in the report
    #tasks.task1()

    # Task 2 runs the second task about EM in the report
    #tasks.task2()

    #Task 3 runs the third task about EM in the report
    #tasks.task3()

    #Task 4 runs the first task about regularized EM in the report
    #To run task 4, line 109 in functions should be replaced by line 108 (different removing criterion)
    #tasks.task4()

    #Task 5 runs the second task about regularized EM in the report
    #Remember to change back to line 109 in functions file
    #tasks.task5()

    #The MNIST code uses PCA with the algorithm from task 5 on the MNIST-data
    MNIST.main()

main()