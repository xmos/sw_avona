// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef SUPPRESSION_WINDOW_H_
#define SUPPRESSION_WINDOW_H_
#include <stdint.h>
#define SUPPRESSION_WINDOW_LENGTH (480)

int32_t sqrt_hanning_480[240];
int32_t rev_wind[240];

void fill_rev_wind(int32_t * rev_wind, int32_t * wind, const unsigned length);

#endif /* SUPPRESSION_WINDOW_H_ */