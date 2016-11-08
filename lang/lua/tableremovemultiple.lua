
local a = {1, 2, 3, 4, 5, 6, 7 }
local b = {1, 2, 3, 4, 5, 6, 7 }

-- remove multiple items from a table
-- @input t is the table to be processed
-- @input idx is a table of indices indicating which one to remove.
--        Note, this function is destructive to this table too.
-- @output None
function tableremovemultiple(t, idx)
   assert(t); assert(idx);
   assert(#idx>0); assert(#t>0);
   local function vecdec(vec)
      for i = 1, #vec do
         vec[i] = vec[i] - 1
      end
   end
   while #idx>0 do
      local cursor = idx[1]
      table.remove(t, cursor)
      table.remove(idx, 1)
      vecdec(idx)
   end
end

print(a)
tableremovemultiple(a, {1,2,3})
print(a)


print(b)
tableremovemultiple(b, {2, 5, 7})
print(b)
