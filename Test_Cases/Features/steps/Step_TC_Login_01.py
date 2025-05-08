# Test_Cases/Features/steps/Step_TC_Login_01.py

from behave import given, when, then
from Common_Functions.CommonFunctions import click, verify_element
from Common_Functions.CommonFunctions import add_screenshot_to_report

@given('the user has launched the ZARA site and rejected cookies')
def step_reject_cookies(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,
                             context.config.userdata["testcase"])
    click(context.driver, 'Reject_Cookies')

@when('the user clicks on the login link')
def step_click_login_link(context):
    click(context.driver, 'login_Menu_Link')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])
    click(context.driver, 'Log_in_Button')

@then('the login form should be displayed')
def step_verify_login_form(context):
    verify_element(context.driver, 'email_field')
    verify_element(context.driver, 'Passcode_field')
    verify_element(context.driver, 'Login_Button')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path,context.config.userdata["testcase"])