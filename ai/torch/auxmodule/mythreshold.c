#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "mythreshold.c"
#else

static int thnnauxmod_(MyThreshold_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  real val = luaT_getfieldchecknumber(L, 1, "val");
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

  if (inPlace) {
    TH_TENSOR_APPLY(real, input,                   \
                    if (*input_data <= threshold) { \
                      *input_data = val;           \
                    });
    THTensor_(set)(output, input);
  } else {
    THTensor_(resizeAs)(output, input);
    TH_TENSOR_APPLY2(real, output, real, input,                         \
                     *output_data = (*input_data > threshold) ? *input_data : val;);

  }

  return 1;
}

static int thnnauxmod_(MyThreshold_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

  if (inPlace) {
    TH_TENSOR_APPLY2(real, gradOutput, real, input,    \
                     if ((*input_data) <= threshold) { \
                       *gradOutput_data = 0;           \
                         });
    THTensor_(set)(gradInput, gradOutput);
  } else {
    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,    \
                     if ((*input_data) > threshold) *gradInput_data = *gradOutput_data; \
                     else *gradInput_data = 0;);                        \
  }

  return 1;
}

static const struct luaL_Reg thnnauxmod_(Threshold__) [] = {
  {"MyThreshold_updateOutput", thnnauxmod_(MyThreshold_updateOutput)},
  {"MyThreshold_updateGradInput", thnnauxmod_(MyThreshold_updateGradInput)},
  {NULL, NULL}
};

static void thnnauxmod_(MyThreshold_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, thnnauxmod_(Threshold__), "thnnauxmod");
  lua_pop(L,1);
}

#endif
