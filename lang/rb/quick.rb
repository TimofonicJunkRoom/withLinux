=begin
https://learnxinyminutes.com/docs/ruby/
$ ruby script.rb
$ irb script.rb
=end

# single line comment

# numbers are objects
3.class
3.to_s

# basic arithmetic
1+1-5*3
35/5
2**5
5%3

# bitwise
3 & 5
3 | 5
3 ^ 5

# arithmetic is syntactic sugar for falling a method on an object
1.+(3)
10.*(5)

# special values are obejcts too
nil
true
false
nil.class

# comparisons and logical operators
1 == 1
2 != 1
!nil # true
!false # true
!0 # false
2 >= 1
1 <=> 10 # -1
10 <=> 1 # 1
1 <=> 1 # 0
true && false
true || false
!true
# do_something() and do_something_else()
# dl_something() or log_error()

# strings
'I am a string'.class
"String".class
placeholder = 'string interpolcation'
"#{placeholder} can be used when using double quoted strings"
'hello' + 'world'
#'hello' + 1 # type error
'hello' + 1.to_s
'hello ' * 3
'hello' << ' world' # append
puts "hello world"
print "hello world"

# variable
x = y = 10
snake_case = true # convention

# symbols
:pending.class
status = :pending

# arrays
arr = [1,2,3,4,5]
arr2 = [1, 'hello', true]
arr[0]
arr.first
arr[100] # nil
arr.[] 0 # arr[0] is just syntactic sugar
arr[-1]
arr.last
arr[2,3] # [start,len]
arr.reverse! # the exclamation means in-place
arr[2..3] # range []
arr << 5
arr.push(6)
arr.include?(1)

# hashes
h = {'color' => 'green', 'number' => 99 }
h.keys
h['color']
h['nil']
h_symbol_as_key = { defcon: 3, action: true }
h_symbol_as_key.key?(:defcon)
h_symbol_as_key.value?(3)

# control
res = if true
	'a'
elsif false
	'b'
else
	'c'
end
res

for counter in 1..5
	puts "iter #{counter}"
end # however no one uses for loops

(1..5).each do |counter|
	puts "iteration #{counter}"
end
(1..5).each { |counter| puts "#{counter}"}
arr.each { |e| puts e}
h.each { |k,v| puts k, v}
arr.each_with_index { |e,idx| puts e, idx }

counter = 1
while counter <= 5 do
	counter += 1
end

doubled = arr.map { |e| e*2 } # see also reduce, inject, and so on

grade = 'B'
case grade
when 'A'
	puts 'A'
else
	puts '???'
end
grade = 80
case grade
when 90..100
	puts 'A'
else
	puts '...'
end

# exception
begin
	raise NoMemoryError, 'out of mem'
rescue NoMemoryError => e
	puts 'raised', e
rescue RuntimeError => other_e
	puts 'raised', other_e
else
	puts 'no exception'
ensure
	puts 'the code is ok'
end

# functions
def double(x)
	x*2
end
double(2)
double 4 # parentheses are optional when not ambiguous
def sum(x, y)
	x+y
end
sum 3,4
sum 1, sum(2, 3)

# yield
def surround
	puts '{'
	yield
	puts '}'
end
surround {puts '  hello'}

# TODO
