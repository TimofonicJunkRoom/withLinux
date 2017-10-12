class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        loc = dict((v, i) for i, v in enumerate(nums)) # O(n)

        for i, v in enumerate(nums): # O(n)
            if loc.get(target-v, False):
                j = loc.get(target-v)
                return [i, j]
        '''
        vtoi = dict()
        for i, v in enumerate(nums):
            #print(i, v, vtoi)
            idx = vtoi.get(target - v, None)
            #print('idx', target-v, idx)
            if None!=idx: return [i, idx]
            else: vtoi[v] = i
        return False;  # O(n), expect O(n/2)

s = Solution()
print(s.twoSum([3,3], 6))
print(s.twoSum([2,7,11,15], 13))
