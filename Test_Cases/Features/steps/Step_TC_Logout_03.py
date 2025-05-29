# ***************************************************************************************************
# File        : Step_TC_Logout_03.py
# Description : Step definitions for logout simulation scenario on ZARA site with AI healing support.
# Author      : Aniket Pathare | Self-Healing AI Framework
# Date        : 2025-05-25
# ***************************************************************************************************

from behave import given, when, then
from Common_Functions.CommonFunctions import click, type_text, verify_element, add_screenshot_to_report

@given('the user is on the ZARA login page')
def user_on_login_page(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'Reject_Cookies')
    click(context.driver, 'login_Menu_Link')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'OAuth_Login_Button')

@when('the user clicks on Logout')
def user_clicks_on_logout(context):
    email = context.testdata.get("Email", "default@example.com")
    pwd = context.testdata.get("Passcode", "defaultPass")
    type_text(context.driver, 'email_field', email)
    type_text(context.driver, 'Password_field', pwd)
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'Login_Button')

@then('the user should be redirected to the ZARA homepage')
def user_redirect_homepage(context):
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
    click(context.driver, 'UserProfileLink')
    click(context.driver, 'Logout_Button')
    add_screenshot_to_report(context.driver, context.report_doc, context.report_path, context.config.userdata["testcase"])
