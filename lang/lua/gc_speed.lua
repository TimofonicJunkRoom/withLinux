-- Allocate, traverse and collect lots of small objects.
-- Shows cache thrashing by the GC. Public domain.

--local N = 14000000
local N = 15000000
local T

local function alloc(text)
  local t = {}
  local x = os.clock()
  for i=1,N do t[i] = {t} end
  print(os.clock()-x, "seconds allocation time"..text)
  T = t
end

local function collect(text)
  x = os.clock()
  collectgarbage()
  print(os.clock()-x, "seconds for a full GC"..text)
  T = nil
  x = os.clock()
  collectgarbage()
  print(os.clock()-x, "seconds for a cleanup GC"..text, "\n")
end

collectgarbage("stop")
alloc(" with stopped GC")
collect("")
collectgarbage("restart")

alloc(" with enabled GC")
collect("")

alloc(" with enabled GC")
local random = math.random
for i=1,N do T[i][1] = T[random(N)] end
collect(" with randomized links")
