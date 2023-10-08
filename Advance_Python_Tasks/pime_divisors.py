from math import sqrt


input_numbers = {}

for _ in range(10):
    input_numbers[int(input("Enter your Number: "))] = 0
   
def is_prime(n):
    
    if (n <= 1):
        return False
    
    for i in range(2, int(sqrt(n))+1):
        if (n % i == 0):
            return False

    return True

def prime_divisors(n):
    count_of_prime_divisors = 0
    
    for i in range(1, int(n/2) + 1):
        if n % i == 0 and is_prime(i):
            count_of_prime_divisors += 1
            
    return count_of_prime_divisors


max_count_of_prime_divisors = (0, 0)

for number in list(input_numbers.keys()):
    
    input_numbers[number] = prime_divisors(number)
    
    if input_numbers[number] > max_count_of_prime_divisors[1]:
        max_count_of_prime_divisors = (number, input_numbers[number])
        continue
    
    elif input_numbers[number] == max_count_of_prime_divisors[1]:
        if number > max_count_of_prime_divisors[0]:
            max_count_of_prime_divisors = (number, input_numbers[number])
            
            
print(max_count_of_prime_divisors[0], max_count_of_prime_divisors[1])