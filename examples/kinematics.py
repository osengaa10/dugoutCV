import time
#
# # z is depth
# # y is height
# # x is lateral movement
#
# x0 = (-95.34040069580078)/1000
# y0 = (257.4190673828125)/1000
# z0 = (842.0)/1000
# time0 = 1618416318381
# x1 = (-97.62313079833984)/1000
# y1 = (257.1994323730469)/1000
# z1 = (829.0)/1000
# time1 = 1618416318403
# x2 = (-98.93209075927734)/1000
# y2 = (245.4981231689453)/1000
# z2 = (809.0)/1000
# time2 = 1618416318437
#
# v0x = (x1 - x0)/((time1 - time0)/1000)
# v0y = (y1 - y0)/((time1 - time0)/1000)
# v0z = (z1 - z0)/((time1 - time0)/1000)
# # v0x = (x1 - x0)/(time1 - time0)
# # v0y = (y1 - y0)/(time1 - time0)
# # v0z = (z1 - z0)/(time1 - time0)
#
# print("v0x: " + str(v0x))
# print("v0y: " + str(v0y))
# print("v0z: " + str(v0z))
#
# v1x = (x2 - x1)/((time2 - time1)/1000)
# v1y = (y2 - y1)/((time2 - time1)/1000)
# v1z = (z2 - z1)/((time2 - time1)/1000)
#
# print("v1x: " + str(v1x))
# print("v1y: " + str(v1y))
# print("v1z: " + str(v1z))
#
# vix = (x2 - x0)/((time2 - time0)/1000)
# viy = (y2 - y0)/((time2 - time0)/1000)
# viz = (z2 - z0)/((time2 - time0)/1000)
#
# print("vix: " + str(vix))
# print("viy: " + str(viy))
# print("viz: " + str(viz))

print(time.monotonic())