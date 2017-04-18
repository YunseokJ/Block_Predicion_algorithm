#include "Block_PRED.h"
#include "Canny_yune.h"
#include <Windows.h>

using namespace cv;
using namespace std;

inline static
void firstScan_Block_PRED(const Mat1b &img, Mat1i& imgLabels, uint* P, uint &lunique) {
	int w(img.cols), h(img.rows);

#define condition_j c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define condition_k c+2<w && r-1>=0 && img_row_prev[c+2]>0
#define condition_i r-1>=0 && img_row_prev[c]>0

#define condition_o img_row[c]>0
#define condition_p c+1<w && img_row[c+1]>0
#define condition_s r+1<h && img_row_fol[c]>0
#define condition_t c+1<w && r+1<h && img_row_fol[c+1]>0
	{
		int r = 0;
		int c = -2;
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
	tree_A0: c += 2; //앞에 background
		if (c >= w) goto break_A0;
		if (condition_o) {
			if (condition_p) {
				imgLabels_row[c] = lunique;
				P[lunique] = lunique; lunique++;
				goto tree_B0;
			}
			else {
				if (condition_t) {
					imgLabels_row[c] = lunique;
					P[lunique] = lunique; lunique++;
					goto tree_B0;
				}
				else {
					imgLabels_row[c] = lunique;
					P[lunique] = lunique; lunique++;
					goto tree_A0;
				}
			}
		}
		else {
			if (condition_p) {
				imgLabels_row[c] = lunique;
				P[lunique] = lunique; lunique++;
				goto tree_B0;
			}
			else {
				if (condition_s) {
					if (condition_t) {
						imgLabels_row[c] = lunique;
						P[lunique] = lunique; lunique++;
						goto tree_B0;
					}
					else {
						imgLabels_row[c] = lunique;
						P[lunique] = lunique; lunique++;
						goto tree_A0;
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique;
						P[lunique] = lunique; lunique++;
						goto tree_B0;
					}
					else {
						imgLabels_row[c] = 0;
						goto tree_A0;
					}
				}
			}
		}
	tree_B0: c += 2; // 앞에 foreground
		if (c >= w) goto break_B0;
		if (condition_o) {
			if (condition_p) {
				imgLabels_row[c] = imgLabels_row[c - 2];
				goto tree_B0;
			}
			else {
				if (condition_t) {
					imgLabels_row[c] = imgLabels_row[c - 2];
					goto tree_B0;
				}
				else {
					imgLabels_row[c] = imgLabels_row[c - 2];
					goto tree_A0;
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
					goto tree_B0;
				}
				else {
					imgLabels_row[c] = lunique;
					P[lunique] = lunique; lunique++;
					goto tree_B0;
				}
			}
			else {
				if (condition_s) {
					if (condition_t) {
						imgLabels_row[c] = imgLabels_row[c - 2];
						goto tree_B0;
					}
					else {
						imgLabels_row[c] = imgLabels_row[c - 2];
						goto tree_A0;
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique;
						P[lunique] = lunique; lunique++;
						goto tree_B0;
					}
					else {
						imgLabels_row[c] = 0;
						goto tree_A0;
					}
				}
			}
		}
	break_A0:
	break_B0:;
	}
	for (int r = 2; r < h; r += 2) {
		int c = 0;
		const uchar* const img_row = img.ptr<uchar>(r);
		const uchar* const img_row_prev = (uchar *)(((char *)img_row) - img.step.p[0]);
		const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);
		uint* const imgLabels_row = imgLabels.ptr<uint>(r);
		uint* const imgLabels_row_prev_prev = (uint *)(((char *)imgLabels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
		if (condition_o) {
			if (condition_p) {
				if (condition_i) {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
							goto tree_E;
						}
						else {
							imgLabels_row[c] = imgLabels_row_prev_prev[c];
							goto tree_D;
						}
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else{
						if (condition_k) {
							imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
							goto tree_E;
						}
						else {
							imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
							goto tree_D;
						}
					}
				}
			}
			else {
				if (condition_i) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						if (condition_t) {
							goto tree_F;
						}
						else {
							goto tree_B;
						}
					}
					else {
						if (condition_t) {
							if (condition_j) {
								goto tree_D;
							}
							else {
								goto tree_G;
							}
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						if (condition_k) {
							if (condition_t) {
								goto tree_F;
							}
							else {
								goto tree_B;
							}
						}
						else {
							if (condition_t) {
								goto tree_G;
							}
							else {
								goto tree_C;
							}
						}
					}
					else {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							if (condition_t) {
								goto tree_F;
							}
							else {
								goto tree_B;
							}
						}
						else {
							if (condition_t) {
								goto tree_D;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_i) {
						if(condition_k){
							imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
							goto tree_E;
						}
						else {
							imgLabels_row[c] = imgLabels_row_prev_prev[c];
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							imgLabels_row[c] = imgLabels_row_prev_prev[c+2];
							goto tree_E;
						}
						else {
							imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
							goto tree_D;
						}
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_A: c += 2; if (c >= w-2 ) goto break_A;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
						goto tree_E;
					}
					else {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						goto tree_D;
					}
				}
			}
			else {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
						goto tree_E;
					}
					else {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						goto tree_D;
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_B: c += 2; if (c >= w-2 ) goto break_B;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) {
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						goto tree_D;
					}
				}
			}
			else {
				imgLabels_row[c] = imgLabels_row_prev_prev[c];
				if (condition_j) {
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { // Q,R
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						goto tree_D;
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_C: c += 2; if (c >= w-2) goto break_C;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) { // P,Q
					imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { //P,R
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else {
						imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
						goto tree_D;
					}
				}
			}
			else {
				if (condition_j) { //PQ
					imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
						goto tree_E;
					}
					else {//
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						goto tree_D;
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_D: c += 2;	if (c >= w-2) goto break_D;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) { // Q,S
					imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { //R,S
						imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else {
						imgLabels_row[c] = imgLabels_row[c - 2];
						goto tree_D;
					}
				}
			}
			else {
				if (condition_j) { //Q,S
					imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					imgLabels_row[c] = imgLabels_row[c - 2];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_s) {
					if (condition_j) { //Q,S
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) { // R,S
							imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
							goto tree_E;
						}
						else {
							imgLabels_row[c] = imgLabels_row[c - 2];
							goto tree_D;
						}
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
							goto tree_E;
						}
						else {
							imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
							goto tree_D;
						}
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_E: c += 2; if (c >= w-2) goto break_E;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) { // Q
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { //Q,R
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						goto tree_D;
					}
				}
			}
			else {
				imgLabels_row[c] = imgLabels_row_prev_prev[c];
				if (condition_j) {
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_j) { //Q
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { // Q,R
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						goto tree_D;
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_F: c += 2; if (c >= w-2) goto break_F;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) { // Q,S
					imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { //Q,R,S
						imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row_prev_prev[c + 2]);
						goto tree_E;
					}
					else { // Q,S
						imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
						goto tree_D;
					}
				}
			}
			else { // Q,S
				imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
				if (condition_j) { //Q,S
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_s) {
					if (condition_j) { //Q,S
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) { // Q,R,S
							imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row_prev_prev[c + 2]);
							goto tree_E;
						}
						else { // Q,S 
							imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
							goto tree_D;
						}
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row_prev_prev[c + 2]);
							goto tree_E;
						}
						else {
							imgLabels_row[c] = imgLabels_row_prev_prev[c];
							goto tree_D;
						}
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	tree_G: c += 2; if (c >= w-2) goto break_G;
		if (condition_o) {
			if (condition_p) {
				if (condition_j) { // P,Q,S
					imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
					if (condition_k) {
						goto tree_E;
					}
					else {
						goto tree_D;
					}
				}
				else {
					if (condition_k) { //P,R,S
						imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c + 2]), imgLabels_row[c - 2]);
						goto tree_E;
					}
					else { //P,S
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
						goto tree_D;
					}
				}
			}
			else {
				if (condition_j) { //P,Q,S
					imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row[c - 2]);
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_G;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_C;
						}
					}
				}
				else {
					imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							goto tree_A;
						}
					}
				}
			}
		}
		else {
			if (condition_p) {
				if (condition_s) {
					if (condition_j) { //Q,S
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) { // R,S
							imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c + 2], imgLabels_row[c - 2]);
							goto tree_E;
						}
						else {
							imgLabels_row[c] = imgLabels_row[c - 2];
							goto tree_D;
						}
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
						if (condition_k) {
							goto tree_E;
						}
						else {
							goto tree_D;
						}
					}
					else {
						if (condition_k) {
							imgLabels_row[c] = imgLabels_row_prev_prev[c + 2];
							goto tree_E;
						}
						else {
							imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
							goto tree_D;
						}
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
					if (condition_t) {
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
						if (condition_k) {
							goto tree_F;
						}
						else {
							if (condition_j) {
								goto tree_G;
							}
							else {
								goto tree_D;
							}
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (condition_k) {
							goto tree_B;
						}
						else {
							if (condition_j) {
								goto tree_C;
							}
							else {
								goto tree_A;
							}
						}
					}
				}
			}
		}
	break_A:
		if (condition_o) {			
			if (condition_j) {
				imgLabels_row[c] = imgLabels_row_prev_prev[c];
			}
			else {
				imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
			}			
		}
		else {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
				}
				else {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	break_B:
		if (condition_o) {
			imgLabels_row[c] = imgLabels_row_prev_prev[c];			
		}
		else {
			if (condition_p) {
				imgLabels_row[c] = imgLabels_row_prev_prev[c];
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	break_C:
		if (condition_o) {
			if (condition_j) {
				imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row_prev_prev[c]);
			}
			else {
				imgLabels_row[c] = imgLabels_row_prev_prev[c - 2];
			}
		}
		else {
			if (condition_p) {
				if (condition_j) {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
				}
				else {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	break_D: //g
		if (condition_o) {
			if (condition_j) {
				imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
			}
			else {
				imgLabels_row[c] = imgLabels_row[c - 2];
			}
		}
		else {
			if (condition_p) {
				if (condition_s) {
					if (condition_j) {
						imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
					}
					else {
						imgLabels_row[c] = imgLabels_row[c - 2];
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
					}
					else {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	break_E: //d
		if (condition_o) {
			imgLabels_row[c] = imgLabels_row_prev_prev[c];
		}
		else {
			if (condition_p) {
				imgLabels_row[c] = imgLabels_row_prev_prev[c];
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c-2];
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	break_F: // e
		if (condition_o) {
			imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
		}
		else {
			if (condition_p) {
				if (condition_s) {
					imgLabels_row[c] = set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]);
				}
				else {
					imgLabels_row[c] = imgLabels_row_prev_prev[c];
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	break_G: // f
		if (condition_o) {
			if (condition_j) {
				imgLabels_row[c] = set_union(P, set_union(P, imgLabels_row[c - 2], imgLabels_row_prev_prev[c]), imgLabels_row_prev_prev[c - 2]);
			}
			else {
				imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c - 2], imgLabels_row[c - 2]);
			}			
		}
		else {
			if (condition_p) {
				if (condition_s) {
					if (condition_j) {
						imgLabels_row[c] = set_union(P, imgLabels_row_prev_prev[c], imgLabels_row[c - 2]);
					}
					else {
						imgLabels_row[c] = imgLabels_row[c - 2];
					}
				}
				else {
					if (condition_j) {
						imgLabels_row[c] = imgLabels_row_prev_prev[c];
					}
					else {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
				}
			}
			else {
				if (condition_s) {
					imgLabels_row[c] = imgLabels_row[c - 2];
				}
				else {
					if (condition_t) {
						imgLabels_row[c] = lunique; P[lunique] = lunique; lunique++;
					}
					else {
						imgLabels_row[c] = 0;
					}
				}
			}
		}
		continue;
	}
}

int Block_PRED(const Mat1b &img, Mat1i &imgLabels) {
	
	imgLabels = cv::Mat1i(img.size());
	//Canny(img, edgeimage, 50, 150);
	//
	//A quick and dirty upper bound for the maximimum number of labels.
	const size_t Plength = img.rows*img.cols / 4;
	//Tree of labels
	uint *P = (uint *)fastMalloc(sizeof(uint)* Plength);
	//Background
	P[0] = 0;
	uint lunique = 1;

	firstScan_Block_PRED(img, imgLabels, P, lunique);

	uint nLabel = flattenL(P, lunique);

	// Second scan
	if (imgLabels.rows & 1) {
		if (imgLabels.cols & 1) {
			//Case 1: both rows and cols odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (c + 1<imgLabels.cols) {
							if (img_row[c + 1]>0)
								imgLabels_row[c + 1] = iLabel;
							else
								imgLabels_row[c + 1] = 0;
							if (r + 1<imgLabels.rows) {
								if (img_row_fol[c]>0)
									imgLabels_row_fol[c] = iLabel;
								else
									imgLabels_row_fol[c] = 0;
								if (img_row_fol[c + 1]>0)
									imgLabels_row_fol[c + 1] = iLabel;
								else
									imgLabels_row_fol[c + 1] = 0;
							}
						}
						else if (r + 1<imgLabels.rows) {
							if (img_row_fol[c]>0)
								imgLabels_row_fol[c] = iLabel;
							else
								imgLabels_row_fol[c] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						if (c + 1<imgLabels.cols) {
							imgLabels_row[c + 1] = 0;
							if (r + 1<imgLabels.rows) {
								imgLabels_row_fol[c] = 0;
								imgLabels_row_fol[c + 1] = 0;
							}
						}
						else if (r + 1<imgLabels.rows) {
							imgLabels_row_fol[c] = 0;
						}
					}
				}
			}
		}//END Case 1
		else {
			//Case 2: only rows odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row[c + 1]>0)
							imgLabels_row[c + 1] = iLabel;
						else
							imgLabels_row[c + 1] = 0;
						if (r + 1<imgLabels.rows) {
							if (img_row_fol[c]>0)
								imgLabels_row_fol[c] = iLabel;
							else
								imgLabels_row_fol[c] = 0;
							if (img_row_fol[c + 1]>0)
								imgLabels_row_fol[c + 1] = iLabel;
							else
								imgLabels_row_fol[c + 1] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row[c + 1] = 0;
						if (r + 1<imgLabels.rows) {
							imgLabels_row_fol[c] = 0;
							imgLabels_row_fol[c + 1] = 0;
						}
					}
				}
			}
		}// END Case 2
	}
	else {
		if (imgLabels.cols & 1) {
			//Case 3: only cols odd
			for (int r = 0; r<imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c]>0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row_fol[c]>0)
							imgLabels_row_fol[c] = iLabel;
						else
							imgLabels_row_fol[c] = 0;
						if (c + 1<imgLabels.cols) {
							if (img_row[c + 1]>0)
								imgLabels_row[c + 1] = iLabel;
							else
								imgLabels_row[c + 1] = 0;
							if (img_row_fol[c + 1]>0)
								imgLabels_row_fol[c + 1] = iLabel;
							else
								imgLabels_row_fol[c + 1] = 0;
						}
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row_fol[c] = 0;
						if (c + 1<imgLabels.cols) {
							imgLabels_row[c + 1] = 0;
							imgLabels_row_fol[c + 1] = 0;
						}
					}
				}
			}
		}// END case 3
		else {
			//Case 4: nothing odd
			for (int r = 0; r < imgLabels.rows; r += 2) {
				// Get rows pointer
				const uchar* const img_row = img.ptr<uchar>(r);
				const uchar* const img_row_fol = (uchar *)(((char *)img_row) + img.step.p[0]);

				uint* const imgLabels_row = imgLabels.ptr<uint>(r);
				uint* const imgLabels_row_fol = (uint *)(((char *)imgLabels_row) + imgLabels.step.p[0]);
				// Get rows pointer
				for (int c = 0; c<imgLabels.cols; c += 2) {
					int iLabel = imgLabels_row[c];
					if (iLabel>0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0)
							imgLabels_row[c] = iLabel;
						else
							imgLabels_row[c] = 0;
						if (img_row[c + 1] > 0)
							imgLabels_row[c + 1] = iLabel;
						else
							imgLabels_row[c + 1] = 0;
						if (img_row_fol[c] > 0)
							imgLabels_row_fol[c] = iLabel;
						else
							imgLabels_row_fol[c] = 0;
						if (img_row_fol[c + 1] > 0)
							imgLabels_row_fol[c + 1] = iLabel;
						else
							imgLabels_row_fol[c + 1] = 0;
					}
					else {
						imgLabels_row[c] = 0;
						imgLabels_row[c + 1] = 0;
						imgLabels_row_fol[c] = 0;
						imgLabels_row_fol[c + 1] = 0;
					}
				}
			}
		}//END case 4
	}

	fastFree(P);
	return nLabel;
}