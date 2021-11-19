(window.webpackJsonp=window.webpackJsonp||[]).push([[39],{606:function(t,e,s){"use strict";s.r(e);var r=s(38),o=Object(r.a)({},(function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h1",{attrs:{id:"builders-domain-scheduling-resource-costs"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-scheduling-resource-costs"}},[t._v("#")]),t._v(" builders.domain.scheduling.resource_costs")]),t._v(" "),s("p"),s("div",{staticClass:"table-of-contents"},[s("ul",[s("li",[s("a",{attrs:{href:"#withmodecosts"}},[t._v("WithModeCosts")]),s("ul",[s("li",[s("a",{attrs:{href:"#get-mode-costs-badge-text-withmodecosts-type-tip"}},[t._v("get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"tip"}})],1)]),s("li",[s("a",{attrs:{href:"#get-mode-costs-badge-text-withmodecosts-type-tip"}},[t._v("_get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"tip"}})],1)])])]),s("li",[s("a",{attrs:{href:"#withoutmodecosts"}},[t._v("WithoutModeCosts")]),s("ul",[s("li",[s("a",{attrs:{href:"#get-mode-costs-badge-text-withmodecosts-type-warn"}},[t._v("get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"warn"}})],1)]),s("li",[s("a",{attrs:{href:"#get-mode-costs-badge-text-withmodecosts-type-warn"}},[t._v("_get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"warn"}})],1)])])]),s("li",[s("a",{attrs:{href:"#withresourcecosts"}},[t._v("WithResourceCosts")]),s("ul",[s("li",[s("a",{attrs:{href:"#get-resource-cost-per-time-unit-badge-text-withresourcecosts-type-tip"}},[t._v("get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"tip"}})],1)]),s("li",[s("a",{attrs:{href:"#get-resource-cost-per-time-unit-badge-text-withresourcecosts-type-tip"}},[t._v("_get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"tip"}})],1)])])]),s("li",[s("a",{attrs:{href:"#withoutresourcecosts"}},[t._v("WithoutResourceCosts")]),s("ul",[s("li",[s("a",{attrs:{href:"#get-resource-cost-per-time-unit-badge-text-withresourcecosts-type-warn"}},[t._v("get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"warn"}})],1)]),s("li",[s("a",{attrs:{href:"#get-resource-cost-per-time-unit-badge-text-withresourcecosts-type-warn"}},[t._v("_get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"warn"}})],1)])])])])]),s("p"),t._v(" "),s("div",{staticClass:"custom-block tip"},[s("p",{staticClass:"custom-block-title"},[t._v("TIP")]),t._v(" "),s("skdecide-summary")],1),t._v(" "),s("h2",{attrs:{id:"withmodecosts"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#withmodecosts"}},[t._v("#")]),t._v(" WithModeCosts")]),t._v(" "),s("p",[t._v("A domain must inherit this class if there are some mode costs to consider.")]),t._v(" "),s("h3",{attrs:{id:"get-mode-costs"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-mode-costs"}},[t._v("#")]),t._v(" get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"tip"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"get_mode_costs",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, float]]"}}}),t._v(" "),s("p",[t._v("Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode\nand the value indicates the cost of execution the task in the mode.")]),t._v(" "),s("h3",{attrs:{id:"get-mode-costs-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-mode-costs-2"}},[t._v("#")]),t._v(" _get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"tip"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"_get_mode_costs",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, float]]"}}}),t._v(" "),s("p",[t._v("Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode\nand the value indicates the cost of execution the task in the mode.")]),t._v(" "),s("h2",{attrs:{id:"withoutmodecosts"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#withoutmodecosts"}},[t._v("#")]),t._v(" WithoutModeCosts")]),t._v(" "),s("p",[t._v("A domain must inherit this class if there are no mode cost to consider.")]),t._v(" "),s("h3",{attrs:{id:"get-mode-costs-3"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-mode-costs-3"}},[t._v("#")]),t._v(" get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"warn"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"get_mode_costs",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, float]]"}}}),t._v(" "),s("p",[t._v("Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode\nand the value indicates the cost of execution the task in the mode.")]),t._v(" "),s("h3",{attrs:{id:"get-mode-costs-4"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-mode-costs-4"}},[t._v("#")]),t._v(" _get_mode_costs "),s("Badge",{attrs:{text:"WithModeCosts",type:"warn"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"_get_mode_costs",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, float]]"}}}),t._v(" "),s("p",[t._v("Return a nested dictionary where the first key is the id of a task (int), the second key the id of a mode\nand the value indicates the cost of execution the task in the mode.")]),t._v(" "),s("h2",{attrs:{id:"withresourcecosts"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#withresourcecosts"}},[t._v("#")]),t._v(" WithResourceCosts")]),t._v(" "),s("p",[t._v("A domain must inherit this class if there are some resource costs to consider.")]),t._v(" "),s("h3",{attrs:{id:"get-resource-cost-per-time-unit"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-resource-cost-per-time-unit"}},[t._v("#")]),t._v(" get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"tip"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"get_resource_cost_per_time_unit",sig:{params:[{name:"self"}],return:"Dict[str, float]"}}}),t._v(" "),s("p",[t._v("Return a dictionary where the key is the name of a resource (str)\nand the value indicates the cost of using this resource per time unit.")]),t._v(" "),s("h3",{attrs:{id:"get-resource-cost-per-time-unit-2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-resource-cost-per-time-unit-2"}},[t._v("#")]),t._v(" _get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"tip"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"_get_resource_cost_per_time_unit",sig:{params:[{name:"self"}],return:"Dict[str, float]"}}}),t._v(" "),s("p",[t._v("Return a dictionary where the key is the name of a resource (str)\nand the value indicates the cost of using this resource per time unit.")]),t._v(" "),s("h2",{attrs:{id:"withoutresourcecosts"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#withoutresourcecosts"}},[t._v("#")]),t._v(" WithoutResourceCosts")]),t._v(" "),s("p",[t._v("A domain must inherit this class if there are no resource cost to consider.")]),t._v(" "),s("h3",{attrs:{id:"get-resource-cost-per-time-unit-3"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-resource-cost-per-time-unit-3"}},[t._v("#")]),t._v(" get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"warn"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"get_resource_cost_per_time_unit",sig:{params:[{name:"self"}],return:"Dict[str, float]"}}}),t._v(" "),s("p",[t._v("Return a dictionary where the key is the name of a resource (str)\nand the value indicates the cost of using this resource per time unit.")]),t._v(" "),s("h3",{attrs:{id:"get-resource-cost-per-time-unit-4"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#get-resource-cost-per-time-unit-4"}},[t._v("#")]),t._v(" _get_resource_cost_per_time_unit "),s("Badge",{attrs:{text:"WithResourceCosts",type:"warn"}})],1),t._v(" "),s("skdecide-signature",{attrs:{name:"_get_resource_cost_per_time_unit",sig:{params:[{name:"self"}],return:"Dict[str, float]"}}}),t._v(" "),s("p",[t._v("Return a dictionary where the key is the name of a resource (str)\nand the value indicates the cost of using this resource per time unit.")])],1)}),[],!1,null,null,null);e.default=o.exports}}]);